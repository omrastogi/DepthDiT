# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import argparse
import gc
import math
import logging
import os
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
from glob import glob

import matplotlib
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from diffusers.models import AutoencoderKL
from omegaconf import OmegaConf
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import wandb


from src.diffusion import create_diffusion
from src.dataset import BaseDepthDataset, DatasetMode, get_dataset
from src.dataset.depth_transform import get_depth_normalizer
from src.dataset.dist_mixed_sampler import DistributedMixedBatchSampler
from src.dataset.hypersim_dataset import HypersimDataset
from src.util.multi_res_noise import multi_res_noise_like
from src.models.models import DiT_models
from pipeline import DepthPipeline

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


#################################################################################
#                             Training Helper Functions                         #
#################################################################################
def cosine_decay_lr_schedule(current_step, warmup_steps, start_step, stop_step, base_lr, final_lr):
    """
    Args:
        current_step (int): The current training step.
        warmup_steps (int): Number of steps for warmup.
        start_step (int): Step at which decay begins.
        stop_step (int): Step at which the scheduler stops and reaches final_lr.
        base_lr (float): The base learning rate (maximum value during training).
        final_lr (float): The final learning rate value at stop_step.

    Returns:
        float: The learning rate at the current step.
    """
    final_lr_ratio = final_lr / base_lr  # Ratio between final and base LR

    if warmup_steps > 0 and current_step < warmup_steps:
        # Warmup phase: linearly increase from 0 to base_lr
        lr = float(current_step) / float(warmup_steps)
    elif current_step < start_step:
        # After warmup but before decay begins: keep at base_lr
        lr = 1.0
    elif current_step >= stop_step:
        # After stop_step: maintain final_lr
        lr = final_lr_ratio
    else:
        # Linearly decay the learning rate multiplier from 1.0 to final_lr_ratio
        decay_steps = stop_step - start_step
        progress = (current_step - start_step) / decay_steps
        lr = 0.5 * (1 + math.cos(math.pi * progress))
    return lr

def linear_decay_lr_schedule(current_step, warmup_steps, start_step, stop_step, base_lr, final_lr):
    """
    Args:
        current_step (int): The current training step.
        warmup_steps (int): Number of steps for warmup.
        start_step (int): Step at which decay begins.
        stop_step (int): Step at which the scheduler stops and reaches final_lr.
        base_lr (float): The base learning rate (maximum value during training).
        final_lr (float): The final learning rate value at stop_step.

    Returns:
        float: The learning rate at the current step.
    """
    final_lr_ratio = final_lr / base_lr  # Ratio between final and base LR

    if warmup_steps > 0 and current_step < warmup_steps:
        # Warmup phase: linearly increase from 0 to base_lr
        lr = float(current_step) / float(warmup_steps)
    elif current_step < start_step:
        # After warmup but before decay begins: keep at base_lr
        lr = 1.0
    elif current_step >= stop_step:
        # After stop_step: maintain final_lr
        lr = final_lr_ratio
    else:
        # Linearly decay the learning rate multiplier from 1.0 to final_lr_ratio
        decay_steps = stop_step - start_step
        lr = 1.0 - ((current_step - start_step) / decay_steps) * (1.0 - final_lr_ratio)
    return lr

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()

#################################################################################
#                             Pipeline Helper Functions                         #
#################################################################################

def create_logger(logging_dir, rank):
    """
    Create a logger that writes to a log file and stdout.
    """
    if rank == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def chw2hwc(chw):
    assert 3 == len(chw.shape)
    if isinstance(chw, torch.Tensor):
        hwc = torch.permute(chw, (1, 2, 0))
    elif isinstance(chw, np.ndarray):
        hwc = np.moveaxis(chw, 0, -1)
    return hwc

def decode_depth(depth_latent, vae):
    """Decode depth latent into depth map"""
    # scale latent
    depth_latent = depth_latent / 0.18215
    # decode
    z = vae.post_quant_conv(depth_latent)
    stacked = vae.decoder(z)
    # mean of output channels
    depth_mean = stacked.mean(dim=1, keepdim=True)
    return depth_mean

def load_config(config_path):
    # Load configuration using OmegaConf
    config = OmegaConf.load(config_path)
    return config

# Modified from Marigold: https://github.com/prs-eth/Marigold/blob/62413d56099d36573b2de1eb8c429839734b7782/src/trainer/marigold_trainer.py#L387
def encode_depth(depth_in, vae):
    # stack depth into 3-channel
    stacked = stack_depth_images(depth_in)
    # encode using VAE encoder
    depth_latent = vae.encode(stacked).latent_dist.sample().mul_(0.18215)
    return depth_latent

def stack_depth_images(depth_in):
    if 4 == len(depth_in.shape):
        stacked = depth_in.repeat(1, 3, 1, 1)
    elif 3 == len(depth_in.shape):
        stacked = depth_in.unsqueeze(1)
        stacked = depth_in.repeat(1, 3, 1, 1)
    return stacked

def colorize_depth_maps(
    depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None
):
    """
    Colorize depth maps.
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored


#################################################################################
#                                  DepthTrainer Class                           #
#################################################################################

class DepthTrainer:
    def __init__(self, args):
        self.args = args

        assert torch.cuda.is_available(), "Training currently requires at least one GPU."

        if dist.is_available():
            dist.init_process_group("nccl")
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.device = self.rank % torch.cuda.device_count()
            self.seed = args.global_seed * self.world_size + self.rank
            torch.cuda.set_device(self.device)
            print(f"Starting rank={self.rank}, seed={self.seed}, world_size={self.world_size}.")
        else:
            dist.init_process_group(backend="nccl", rank=0, world_size=1)
            self.world_size = 1
            self.rank = 0
            self.device = 0
            self.seed = args.global_seed
            print(f"Running on single GPU. seed={self.seed}")

        # Set seed for reproducibility
        torch.manual_seed(self.seed)

        self.config = load_config(self.args.config_path)

        # Setup an experiment folder:
        if self.rank == 0:
            os.makedirs(args.results_dir, exist_ok=True)
            experiment_index = len(glob(f"{args.results_dir}/*"))
            model_string_name = args.model.replace("/", "-")
            experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-{args.training_label}--{datetime.now().strftime('%m%d-%H:%M:%S')}"
            checkpoint_dir = f"{experiment_dir}/checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.experiment_dir = experiment_dir
            self.checkpoint_dir = checkpoint_dir
            self.logger = create_logger(experiment_dir, self.rank)
            self.logger.info(f"Experiment directory created at {experiment_dir}")
        else:
            self.experiment_dir = None
            self.checkpoint_dir = None
            self.logger = create_logger(None, self.rank)

        # Initialize WandB only if rank == 0 (main process)
        if self.rank == 0:
            wandb.init(
                project="DiT-Onestep-Training",  # Replace with your WandB project name
                name=f"{args.model}-{args.image_size}-{args.training_label}-{datetime.now().strftime('%m%d-%H:%M:%S')}",
                config=vars(args),
            )

        self.batch_size = int(args.global_batch_size // self.world_size)
        self.logger.info(f"Batch size: {self.batch_size}")

        # Initialize model, ema, optimizer, etc.
        self.create_and_initialize_model()
        # Create datasets
        self.create_datasets()

        # Prepare DDP
        if dist.is_available() and dist.is_initialized():
            self.model = DDP(self.model.to(self.device), device_ids=[self.device])
        else:
            self.model = self.model.to(self.device)

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if self.optimizer_dict is not None:
            self.opt.load_state_dict(self.optimizer_dict)
            
        self.lr_scheduler = LambdaLR(
            self.opt,
            lr_lambda=lambda step: cosine_decay_lr_schedule(
                step, 
                warmup_steps=self.config.lr_scheduler.warmup, 
                start_step=self.config.lr_scheduler.start_step, 
                stop_step=self.config.lr_scheduler.stop_step, 
                base_lr=args.lr, 
                final_lr=self.config.lr_scheduler.stable_lr),
            )

        # # Start training
        # self.train()

    def _replace_patchembed_proj(self):
        # Replace the first layer to accept 8 in_channels
        _weight = self.model.x_embedder.proj.weight.clone()  # [320, 4, 3, 3]
        _bias = self.model.x_embedder.proj.bias.clone()  # [320]
        _weight = _weight.repeat((1, 2, 1, 1))  # Keep selected channel(s)
        # Half the activation magnitude
        _weight *= 0.5
        # New proj channel
        _n_proj_out_channel = self.model.x_embedder.proj.out_channels
        kernel_size = self.model.x_embedder.proj.kernel_size
        padding = self.model.x_embedder.proj.padding
        stride = self.model.x_embedder.proj.stride
        _new_proj = Conv2d(
            8, _n_proj_out_channel, kernel_size=kernel_size, stride=stride, padding=padding
        )
        _new_proj.weight = Parameter(_weight)
        _new_proj.bias = Parameter(_bias)
        self.model.x_embedder.proj = _new_proj
        print("PatchEmbed proj layer is replaced")

    def create_and_initialize_model(self):
        """
        Create and initialize the model, EMA model, and other components.
        Assigns them directly to class variables.
        """
        # Create model:
        assert self.args.image_size % 8 == 0, "Image size must be divisible by 8."
        latent_size = self.args.image_size // 8
        self.model = DiT_models[self.args.model](input_size=latent_size, num_classes=self.args.num_classes)

        # Load the checkpoint
        checkpoint = torch.load(self.args.pretrained_path, map_location=lambda storage, loc: storage)

        if "model" in checkpoint:  # Check if it's a checkpoint saved during training
            state_dict_model = checkpoint["model"]
        else:
            state_dict_model = checkpoint

        # Check if the checkpoint is for a Depth-DiT model
        if "is_depth" in checkpoint:
            # The state_dict is already modified for depth, so modify the model first
            self._replace_patchembed_proj()
            # Load the state_dict into the model
            self.model.load_state_dict(state_dict_model)
        else:
            # For a vanilla DiT model:
            # Load the state_dict first, and then modify the model afterwards
            self.model.load_state_dict(state_dict_model)
            self._replace_patchembed_proj()

        # Create EMA model and set it as non-trainable
        self.ema = deepcopy(self.model).to(self.device)  # EMA model
        requires_grad(self.ema, False)

        # If the checkpoint has the EMA state, load it
        if "ema" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema"])
            self.logger.info("EMA state loaded from checkpoint.")

        # Initialize the optimizer only if the optimizer state is present in the checkpoint
        if "iteration" in checkpoint:
            self.train_step = checkpoint["iteration"]
        else:
            self.train_step = 0

        if "opt" in checkpoint:
            self.optimizer_dict = checkpoint["opt"]
        else:
            self.optimizer_dict = None

        self.diffusion = create_diffusion(timestep_respacing="")
        self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{self.args.vae}").to(self.device)

        self.logger.info(f"DiT Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def create_datasets(self):
        loader_seed = self.args.global_seed
        if loader_seed is None:
            loader_generator = None
        else:
            loader_generator = torch.Generator().manual_seed(loader_seed)

        cfg = self.config

        # Training dataset
        depth_transform = get_depth_normalizer(
            cfg_normalizer=cfg.depth_normalization
        )

        train_dataset: BaseDepthDataset = get_dataset(
            cfg.dataset.train,
            base_data_dir=cfg.paths.base_data_dir,
            mode=DatasetMode.TRAIN,
            augmentation_args=cfg.augmentation,
            depth_transform=depth_transform,
        )

        if "mixed" == cfg.dataset.train.name:
            dataset_ls = train_dataset
            assert len(cfg.dataset.train.prob_ls) == len(
                dataset_ls
            ), "Lengths don't match: `prob_ls` and `dataset_list`"
            concat_dataset = ConcatDataset(dataset_ls)

            self.sampler = DistributedMixedBatchSampler(
                src_dataset_ls=dataset_ls,
                batch_size=self.batch_size,
                shuffle=True,
                world_size=self.world_size,
                rank=self.rank,
                prob=cfg.dataset.train.prob_ls,
                generator=loader_generator,
                drop_last=True,
            )

            self.train_loader = DataLoader(
                concat_dataset,
                batch_sampler=self.sampler,
                num_workers=self.args.num_workers,
            )
        else:
            self.sampler = DistributedSampler(
                train_dataset, 
                num_replicas=self.world_size, 
                rank=self.rank, 
                shuffle=True, 
                seed=self.args.global_seed,
                drop_last=True,
            )

            self.train_loader = DataLoader(
                dataset=train_dataset,
                sampler=self.sampler,
                batch_size=self.batch_size,
                num_workers=self.args.num_workers,
                generator=loader_generator,
            )
            
        val_dataset = HypersimDataset(
            mode=DatasetMode.EVAL,  # Since TRAIN changes the shape
            filename_ls_path=cfg.dataset.val.filenames,
            dataset_dir=os.path.join(cfg.paths.base_data_dir, cfg.dataset.val.dir),
            resize_to_hw=cfg.dataset.val.resize_to_hw,
            disp_name=cfg.dataset.val.name,
            depth_transform=depth_transform
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.validation.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

    def validation(self):
        print("Validation started")
        pipe = DepthPipeline(
            batch_size=self.config.validation.batch_size,
            cfg_scale=self.config.validation.cfg_scale,
            ensemble_size=1,
            num_sampling_steps=self.config.validation.diffusion_steps,
            scheduler=self.config.validation.scheduler,
            model=self.model.module,
            vae=self.vae,
            diffusion=create_diffusion(str(self.config.validation.diffusion_steps))
        )
        depth_colored, images = [], []
        for batch in self.val_loader:
            # depth_pred, depth_colored_img, pred_uncert
            rgb_int = batch["rgb_int"].to(self.device)
            for rgb in range(rgb_int.shape[0]):
                image = Image.fromarray(rgb_int[rgb].cpu().numpy().transpose(1, 2, 0).astype(np.uint8))
                depth_pred, depth_colored_img, pred_uncert = pipe.pipe(image) 
                depth_colored.append(depth_colored_img)
                images.append(image)
        
        # Log depth images and real images to wandb
        wandb_images = []
        for i, depth_colored_img in enumerate(depth_colored):
            wandb_images.append(wandb.Image(depth_colored_img, caption=f"Depth Image {i}"))

        # Also log real images from rgb_int
        for i, rgb in enumerate(images):
            wandb_images.append(wandb.Image(rgb, caption=f"Real Image {i}"))

        # Log to wandb
        if self.rank == 0:
            wandb.log({f"validation_images_step_{self.train_step}": wandb_images, "step": self.train_step})

        print("Validation completed and images logged to wandb.")
        gc.collect()
        torch.cuda.empty_cache()

    def train(self):

        # Prepare models for training:
        update_ema(self.ema, self.model.module, decay=0)  # Initialize EMA
        self.model.train()
        self.ema.eval()

        # Variables for monitoring/logging purposes:
        log_steps, running_loss = 0, 0
        epoch = 0
        self.logger.info(f"Training for {self.args.iterations} iterations...")

        train_loader_iter = iter(self.train_loader)

        for train_step in tqdm(range(self.train_step, self.args.iterations), initial=self.train_step, total=self.args.iterations, desc="Training Progress"):
            if train_step % len(self.train_loader) == 0:
                epoch += 1
                self.sampler.set_epoch(epoch)
                train_loader_iter = iter(self.train_loader)

            batch = next(train_loader_iter)

            rgb = batch["rgb_norm"].to(self.device)
            depth_gt_for_latent = batch['depth_raw_norm'].to(self.device)
            if self.args.valid_mask_loss:
                valid_mask_for_latent = batch['valid_mask_raw'].to(self.device)
                invalid_mask = ~valid_mask_for_latent
                valid_mask_down = ~torch.max_pool2d(invalid_mask.float(), 8, 8).bool().repeat((1, 4, 1, 1))
            else:
                valid_mask_down = None

            with torch.no_grad():
                rgb_input_latent = self.vae.encode(rgb).latent_dist.sample().mul_(0.18215)
                x = encode_depth(depth_gt_for_latent, self.vae)

            y = torch.zeros(self.batch_size, dtype=torch.long).to(self.device)
            timesteps = torch.randint(0, self.diffusion.num_timesteps, (x.shape[0],), device=self.device)

            if self.config.multi_res_noise is not None:
                strength = self.config.multi_res_noise.strength
                if self.config.multi_res_noise.annealing:
                    # Calculate strength depending on t
                    strength = strength * (timesteps / self.diffusion.num_timesteps)
                    noise = multi_res_noise_like(
                        x,
                        strength=strength,
                        downscale_strategy=self.config.multi_res_noise.downscale_strategy,
                        device=self.device,
                    )
            else:
                noise = None

            model_kwargs = dict(y=y, input_img=rgb_input_latent)
            loss_dict = self.diffusion.training_losses(self.model, x, timesteps, model_kwargs, valid_mask_down, noise)
            loss = loss_dict["loss"].mean()

            self.opt.zero_grad()
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            loss.backward()
            self.opt.step()
            self.lr_scheduler.step()
            update_ema(self.ema, self.model.module)

            # Logging
            running_loss += loss.item()
            log_steps += 1
            train_step += 1
            current_lr = self.lr_scheduler.get_last_lr()[0]
            avg_loss = running_loss / log_steps
            if self.rank == 0:
                wandb.log(
                    {
                    "smooth_loss": avg_loss, 
                    "train_loss": loss.item(), 
                    "step": train_step, 
                    "epoch": epoch, 
                    "lr":current_lr
                    }
                )

            # Validation
            if (train_step % self.args.validation_every == 0 or train_step in [50, 100, 250, 500, 750, 1000]) and self.rank == 0:
                self.validation()

            # Save checkpoints:
            if train_step % self.args.ckpt_every == 0 and train_step > 0:
                if self.rank == 0:
                    checkpoint = {
                        "model": self.model.module.state_dict(),
                        "ema": self.ema.state_dict(),
                        "opt": self.opt.state_dict(),
                        "args": self.args,
                        "is_depth": True,
                        "iteration": train_step
                    }
                    checkpoint_path = f"{self.checkpoint_dir}/{train_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    self.logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

        self.logger.info("Training complete!")
        cleanup()

    # def validate_single_dataset(
    #     self,
    #     data_loader: DataLoader,
    #     metric_tracker: MetricTracker,
    #     save_to_dir: str = None,
    # ):
    #     self.model.to(self.device)
    #     metric_tracker.reset()

    #     # Generate seed sequence for consistent evaluation
    #     val_init_seed = self.cfg.validation.init_seed
    #     val_seed_ls = generate_seed_sequence(val_init_seed, len(data_loader))

    #     for i, batch in enumerate(
    #         tqdm(data_loader, desc=f"evaluating on {data_loader.dataset.disp_name}"),
    #         start=1,
    #     ):
    #         assert 1 == data_loader.batch_size
    #         # Read input image
    #         rgb_int = batch["rgb_int"].squeeze()  # [3, H, W]
    #         # GT depth
    #         depth_raw_ts = batch["depth_raw_linear"].squeeze()
    #         depth_raw = depth_raw_ts.numpy()
    #         depth_raw_ts = depth_raw_ts.to(self.device)
    #         valid_mask_ts = batch["valid_mask_raw"].squeeze()
    #         valid_mask = valid_mask_ts.numpy()
    #         valid_mask_ts = valid_mask_ts.to(self.device)

    #         # Random number generator
    #         seed = val_seed_ls.pop()
    #         if seed is None:
    #             generator = None
    #         else:
    #             generator = torch.Generator(device=self.device)
    #             generator.manual_seed(seed)

    #         # Predict depth
    #         pipe_out: MarigoldDepthOutput = self.model(
    #             rgb_int,
    #             denoising_steps=self.cfg.validation.denoising_steps,
    #             ensemble_size=self.cfg.validation.ensemble_size,
    #             processing_res=self.cfg.validation.processing_res,
    #             match_input_res=self.cfg.validation.match_input_res,
    #             generator=generator,
    #             batch_size=1,  # use batch size 1 to increase reproducibility
    #             color_map=None,
    #             show_progress_bar=False,
    #             resample_method=self.cfg.validation.resample_method,
    #         )

    #         depth_pred: np.ndarray = pipe_out.depth_np

    #         if "least_square" == self.cfg.eval.alignment:
    #             depth_pred, scale, shift = align_depth_least_square(
    #                 gt_arr=depth_raw,
    #                 pred_arr=depth_pred,
    #                 valid_mask_arr=valid_mask,
    #                 return_scale_shift=True,
    #                 max_resolution=self.cfg.eval.align_max_res,
    #             )
    #         else:
    #             raise RuntimeError(f"Unknown alignment type: {self.cfg.eval.alignment}")

    #         # Clip to dataset min max
    #         depth_pred = np.clip(
    #             depth_pred,
    #             a_min=data_loader.dataset.min_depth,
    #             a_max=data_loader.dataset.max_depth,
    #         )

    #         # clip to d > 0 for evaluation
    #         depth_pred = np.clip(depth_pred, a_min=1e-6, a_max=None)

    #         # Evaluate
    #         sample_metric = []
    #         depth_pred_ts = torch.from_numpy(depth_pred).to(self.device)

    #         for met_func in self.metric_funcs:
    #             _metric_name = met_func.__name__
    #             _metric = met_func(depth_pred_ts, depth_raw_ts, valid_mask_ts).item()
    #             sample_metric.append(_metric.__str__())
    #             metric_tracker.update(_metric_name, _metric)

    #         # Save as 16-bit uint png
    #         if save_to_dir is not None:
    #             img_name = batch["rgb_relative_path"][0].replace("/", "_")
    #             png_save_path = os.path.join(save_to_dir, f"{img_name}.png")
    #             depth_to_save = (pipe_out.depth_np * 65535.0).astype(np.uint16)
    #             Image.fromarray(depth_to_save).save(png_save_path, mode="I;16")

    #     return metric_tracker.result()

if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=512)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--validation-every", type=int, default=200)
    parser.add_argument("--valid-mask-loss", action="store_true", help="Use valid mask loss")
    parser.add_argument("--pretrained-path", type=str, default="checkpoints/DiT-XL-2-512x512.pt", help="Path to the pretrained model checkpoint")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--weight-decay", type=float, default=0, help="Weight decay for the optimizer")
    parser.add_argument("--training-label", type=str, default="training", help="Label for the training session")
    parser.add_argument("--config-path", type=str, default="config/training_config.yaml", help="Path of configuration script")
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    trainer = DepthTrainer(args)
    trainer.train()

'''
torchrun --nnodes=1 --nproc_per_node=1 train_depth.py --model DiT-XL/2 --epochs 20 --validation-every 5 --global-batch-size 4 --ckpt-every 20 --image-size 512 --data-path /mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/DiT/data/imagenet/train
'''

# For two gpus
'''
torchrun --nnodes=1 --nproc_per_node=2  train_depth.py \
--model DiT-XL/2 \
--valid-mask-loss \
--validation-every 500 \
--iteration 20000 \
--global-batch-size 20 \
--ckpt-every 2000 \
--config-path config/training_config.yaml \
--results-dir checkpoints
'''


'''
torchrun --nnodes=1 --nproc_per_node=2  train_depth.py \
--model DiT-XL/2 \
--valid-mask-loss \
--validation-every 1 \
--global-batch-size 4 \
--ckpt-every 5 \
--image-size 512 \
'''


# OVERFITTING
'''
torchrun --nnodes=1 --nproc_per_node=2  train_depth.py \
--model DiT-XL/2 \
--valid-mask-loss \
--validation-every 500 \
--iteration 5000 \
--global-batch-size 20 \
--ckpt-every 5000 \
--config-path config/overfitting_config.yaml \
--results-dir checkpoints
'''

# TESTING
"""
Depth-DiT and Vanilla-DiT checkpoints 
- Run with Vanilla-DiT model - Done
- Run with Depth-DiT model - Done - wandb log is very close to the validation outputs at that step  
"""

# TODO

"""
Add support to open both Depth-DiT and Vanilla-DiT checkpoints - DONE 
"""
"""
Test the wandb label
"""

"""
Add Multi-resolution noise 
- First document all that is happening in Marigold's training 
- Then create the multi-res noise here
"""
"""
LOWER PRIORITY 
Make sure that the ema is being updated properly. 
- See if its required or just redundant 
- If required, 
- - make sure it updated
- - use it for val 
- - use it for inference 
"""