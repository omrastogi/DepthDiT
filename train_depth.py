# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
from torch.nn import Conv2d
from torch.nn.utils import clip_grad_norm_
from torch.nn.parameter import Parameter
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import gc

import numpy as np 
import matplotlib
from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from tqdm import tqdm
import wandb

from dataset.base_depth_dataset import BaseDepthDataset, get_pred_name, DatasetMode  # noqa: F401
from dataset.hypersim_dataset import HypersimDataset
from dataset.depth_transform import get_depth_normalizer
from types import SimpleNamespace


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

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


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
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

#################################################################################
#                                  Depth Related Function                       #
#################################################################################


def _replace_patchembed_proj(model):
    # https://github.com/prs-eth/Marigold/blob/518ab83a328ecbf57e7d63ec370e15dfa4336598/src/trainer/marigold_trainer.py#L169
    # replace the first layer to accept 8 in_channels
    _weight = model.x_embedder.proj.weight.clone()  # [320, 4, 3, 3]
    _bias = model.x_embedder.proj.bias.clone()  # [320]
    _weight = _weight.repeat((1, 2, 1, 1))  # Keep selected channel(s)
    # half the activation magnitude
    _weight *= 0.5
    # new proj channel
    _n_proj_out_channel = model.x_embedder.proj.out_channels
    kernel_size=model.x_embedder.proj.kernel_size
    padding=model.x_embedder.proj.padding
    stride=model.x_embedder.proj.stride
    _new_proj = Conv2d(
        8, _n_proj_out_channel, kernel_size=kernel_size, stride=stride, padding=padding
    )
    _new_proj.weight = Parameter(_weight)
    _new_proj.bias = Parameter(_bias)
    model.x_embedder.proj = _new_proj
    print("PatchEmbed proj layer is replaced")
    # replace config - Not required for DiT
    # self.model.unet.config["in_channels"] = 8
    # print("Unet config is updated")
    return model

#################################################################################
#                                  Training Loop                                #
#################################################################################

def validation(model, loader, vae, device, step, rank):
    print("Validation started")
    diffusion = create_diffusion(str(25))
    
    # Get the next batch
    batch = next(iter(loader))
    rgb = batch["rgb_norm"].to(device)
    rgb_int = batch["rgb_int"].to(device)  # Real RGB images from batch

    # mentioning params
    batch_size = rgb.shape[0]
    cfg_scale = 4.0 # TODO: remove the hardcoding
    
    # Zero the class-conditioning
    y = torch.zeros(batch_size, dtype=torch.long).to(device)
    
    with torch.no_grad(): # Map input images to latent space + normalize latents:
        rgb_input_latent = vae.encode(rgb).latent_dist.sample().mul_(0.18215)
    
    noise = torch.randn_like(rgb_input_latent, device=device)

    # Setup classifier-free guidance:
    noise = torch.cat([noise, noise], 0)
    y_null = torch.tensor([1000] * batch_size, device=device)
    y = torch.cat([y, y_null], 0)
    rgb_input_latent = torch.cat([rgb_input_latent, rgb_input_latent], 0) # concating this as well
    model_kwargs = dict(y=y, cfg_scale=cfg_scale, input_img=rgb_input_latent)

    # Sample images using the diffusion model
    samples = diffusion.p_sample_loop(
        model.module.forward_with_cfg, noise.shape, noise, 
        clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )

    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    
    # Decode the depth from latent space
    depth = decode_depth(samples, vae)
    depth = torch.clip(depth, -1.0, 1.0)
    
    # Normalize depth values between 0 and 1
    depth_pred = (depth + 1.0) / 2.0
    depth_pred = depth_pred.squeeze()
    
    # Convert depth prediction to numpy array
    depth_pred = depth_pred.detach().cpu().numpy()
    depth_pred = depth_pred.clip(0, 1)
    
    # Colorize depth maps using a colormap
    depth_colored = colorize_depth_maps(depth_pred, 0, 1, cmap="Spectral").squeeze()
    
    # Convert to uint8 for wandb logging
    depth_colored = (depth_colored * 255).astype(np.uint8)
    
    # Log depth images and real images to wandb
    wandb_images = []
    for i in range(depth_colored.shape[0]):
        depth_colored_hwc = chw2hwc(depth_colored[i])
        # Log depth image to wandb
        wandb_images.append(wandb.Image(depth_colored_hwc, caption=f"Depth Image {i}"))

    # Also log real images from rgb_int
    rgb_int_np = rgb_int.detach().cpu().numpy()
    for i in range(rgb_int_np.shape[0]):
        real_image_hwc = chw2hwc(rgb_int_np[i])
        wandb_images.append(wandb.Image(real_image_hwc, caption=f"Real Image {i}"))

    # Log to wandb
    if rank == 0:
        wandb.log({f"validation_images_step_{step}": wandb_images, "step": step})

    # # Log as tabular in wandb
    # table = wandb.Table(columns=["Real Image", "Depth Image"])

    # # Get the numpy arrays for real images and depth images
    # rgb_int_np = rgb_int.detach().cpu().numpy()
    # depth_colored_np = depth_colored  # Already in numpy uint8 format

    # # For each image in the batch
    # for i in range(depth_colored_np.shape[0]):
    #     # Convert the real image to HWC format
    #     real_image_hwc = chw2hwc(rgb_int_np[i])
    #     # Convert the depth image to HWC format
    #     depth_colored_hwc = chw2hwc(depth_colored_np[i])
        
    #     # Create wandb.Image objects
    #     real_image = wandb.Image(real_image_hwc, caption=f"Real Image {i}")
    #     depth_image = wandb.Image(depth_colored_hwc, caption=f"Depth Image {i}")
        
    #     # Add a row to the table with the real image and depth image
    #     table.add_data(real_image, depth_image)

    # # Log the table to wandb
    # wandb.log({f"validation_images_step_{step}": table, "step": step})


    print("Validation completed and images logged to wandb.")
    gc.collect()
    torch.cuda.empty_cache()



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


def create_and_initialize_model(args, device, rank, logger):
    """
    Create and initialize the model, EMA model, and other components.
    Args:
        args: Arguments containing model parameters and paths.
        device: The device (CPU or GPU) to load the model on.
        rank: The rank for DistributedDataParallel (DDP).
        logger: Logger for logging information.
    Returns:
        Tuple containing the model, EMA model, diffusion, VAE, and optimizer (if available).
    """
    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](input_size=latent_size, num_classes=args.num_classes)

    # Load the checkpoint
    checkpoint = torch.load(args.pretrained_path, map_location=lambda storage, loc: storage)

    if "model" in checkpoint:  # Check if it's a checkpoint saved during training
        state_dict_model = checkpoint["model"]
    else:
        state_dict_model = checkpoint

    # Check if the checkpoint is for a Depth-DiT model
    if "is_depth" in checkpoint:
        # The state_dict is already modified for depth, so modify the model first
        model = _replace_patchembed_proj(model)
        # Load the state_dict into the model
        model.load_state_dict(state_dict_model)
    else:
        # For a vanilla DiT model:
        # Load the state_dict first, and then modify the model afterwards
        model.load_state_dict(state_dict_model)
        model = _replace_patchembed_proj(model)
    
    # Create EMA model and set it as non-trainable
    ema = deepcopy(model).to(device)  # EMA model
    requires_grad(ema, False)

    # If the checkpoint has the EMA state, load it
    if "ema" in checkpoint:
        ema.load_state_dict(checkpoint["ema"])
        logger.info("EMA state loaded from checkpoint.")

    # Initialize the optimizer only if the optimizer state is present in the checkpoint
    optimizer_dict = None
    if "opt" in checkpoint:
        optimizer_dict = checkpoint["opt"]
        # Define the optimizer (make sure to match this with your training setup)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # optimizer.load_state_dict(checkpoint["opt"])
        # logger.info("Optimizer state loaded from checkpoint.")
    
    diffusion = create_diffusion(timestep_respacing="")
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, ema, diffusion, vae, optimizer_dict


def main(args):
    """Trains a new DiT model"""
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Initialize WandB only if rank == 0 (main process)
    if rank == 0:
        wandb.init(
            project="DiT-Training",  # Replace with your WandB project name
            name=f"DiT-{args.model}-{args.image_size}x{args.image_size}_code_testing",
            config=vars(args),
        )

    model, ema, diffusion, vae, optimizer_dict = create_and_initialize_model(args, device, rank, logger)
    model = DDP(model.to(device), device_ids=[rank])

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if optimizer_dict is not None:
        opt.load_state_dict(optimizer_dict)

    
    #TODO: Read config.yaml 
    # Setup dataset:
    cfg_data_split = {
        'name': 'hypersim',
        'disp_name': 'hypersim_train',
        'dir': 'Hypersim/processed/train',
        'filenames': 'data_split/hypersim/filename_list_train_filtered.txt',
        'resize_to_hw': [512, 512],
    }

    depth_transform = get_depth_normalizer(SimpleNamespace(
        type='scale_shift_depth', clip=True, norm_min=-1.0, norm_max=1.0, min_max_quantile=0.02))
    kwargs = {'augmentation_args': {'lr_flip_p': 0.5}, 'depth_transform': depth_transform}
    kwargs['augmentation_args'] = SimpleNamespace(**kwargs['augmentation_args'])

    train_dataset = HypersimDataset(
        mode=DatasetMode.TRAIN,
        filename_ls_path=cfg_data_split['filenames'],
        dataset_dir=os.path.join("/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/Marigold/data", cfg_data_split['dir']),
        **cfg_data_split,
        **kwargs,
    )

    val_dataset = HypersimDataset(
        mode=DatasetMode.TRAIN, # since TRAIN changes the shape
        filename_ls_path="/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/DiT/data_split/hypersim/selected_vis_sample.txt",
        dataset_dir=os.path.join("/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/Marigold/data", "Hypersim/processed/val"),
        **cfg_data_split,
        **kwargs,
    )


    bsz = int(args.global_batch_size // dist.get_world_size())

    sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=True, seed=args.global_seed)
    train_loader = DataLoader(train_dataset, batch_size=bsz, sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    val_loader = DataLoader(val_dataset, batch_size=3, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    logger.info(f"Dataset contains {len(train_dataset):,} images")
    logger.info(f"Batch size: {bsz}")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Initialize EMA
    model.train()
    ema.eval()

    # Variables for monitoring/logging purposes:
    train_steps, log_steps, running_loss = 0, 0, 0

    logger.info(f"Training for {args.epochs} epochs...")
    
    # if rank == 0:
    #     validation(model, val_loader, vae, device, train_steps, rank)
    
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            # Training step:
            rgb = batch["rgb_norm"].to(device)
            depth_gt_for_latent = batch['depth_raw_norm'].to(device)

            if args.valid_mask_loss:
                valid_mask_for_latent = batch['valid_mask_raw'].to(device)
                invalid_mask = ~valid_mask_for_latent
                valid_mask_down = ~torch.max_pool2d(invalid_mask.float(), 8, 8).bool().repeat((1, 4, 1, 1))
            else:
                valid_mask_down = None

            y = torch.zeros(bsz, dtype=torch.long).to(device)

            with torch.no_grad():
                rgb_input_latent = vae.encode(rgb).latent_dist.sample().mul_(0.18215)
                x = encode_depth(depth_gt_for_latent, vae)

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y, input_img=rgb_input_latent)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs, valid_mask_down) 
            loss = loss_dict["loss"].mean()

            opt.zero_grad()

            clip_grad_norm_(model.parameters(), max_norm=1.0)
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Logging
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            avg_loss = running_loss / log_steps
            if rank == 0:
                wandb.log({"train_loss": avg_loss, "step": train_steps, "epoch": epoch})

            # Validation
            if train_steps % args.validation_every == 0 and rank == 0: 
                validation(model, val_loader, vae, device, train_steps, rank)
            
            # Save checkpoints:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "is_depth": True
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    logger.info("Training complete!")
    cleanup()

# Modified from Marigold: https://github.com/prs-eth/Marigold/blob/62413d56099d36573b2de1eb8c429839734b7782/src/trainer/marigold_trainer.py#L387
def encode_depth(depth_in, vae):
    # stack depth into 3-channel
    stacked = stack_depth_images(depth_in)
    # encode using VAE encoder
    depth_latent = vae.encode(stacked).latent_dist.sample().mul_(0.18215)
    return depth_latent

@staticmethod
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

if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--validation-every", type=int, default=200)
    parser.add_argument("--valid-mask-loss", action="store_true", help="Use valid mask loss")
    parser.add_argument("--pretrained-path", type=str, default="/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/DiT/checkpoints/DiT-XL-2-512x512.pt", help="Path to the pretrained model checkpoint")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--weight-decay", type=float, default=0, help="Weight decay for the optimizer")
    args = parser.parse_args()
    main(args)

'''
torchrun --nnodes=1 --nproc_per_node=1 train_depth.py --model DiT-XL/2 --epochs 20 --validation-every 5 --global-batch-size 4 --ckpt-every 20 --image-size 512 --data-path /mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/DiT/data/imagenet/train
'''

# For two gpus
'''
torchrun --nnodes=1 --nproc_per_node=2  train_depth.py \
--model DiT-XL/2 \
--valid-mask-loss \
--epochs 6 \
--validation-every 1000 \
--global-batch-size 20 \
--ckpt-every 2000 \
--image-size 512 \
--pretrained-path /mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/DiT/results/003-DiT-XL-2/checkpoints/0012000.pt \
--data-path /mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/DiT/data/imagenet/train
'''


'''
torchrun --nnodes=1 --nproc_per_node=1  train_depth.py \
--model DiT-XL/2 \
--valid-mask-loss \
--epochs 1 \
--validation-every 1 \
--global-batch-size 4 \
--ckpt-every 5 \
--image-size 512 \
--pretrained-path /mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/DiT/results/003-DiT-XL-2/checkpoints/0012000.pt \
--data-path /mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/DiT/data/imagenet/train
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