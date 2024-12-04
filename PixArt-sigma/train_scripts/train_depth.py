import gc
import os
import sys
import math
import time
import types
import wandb
import warnings
import argparse
import datetime
from tqdm import tqdm
import torchvision.transforms.functional as F
import torchvision.transforms as T
from pathlib import Path
from copy import deepcopy
from omegaconf import OmegaConf
from collections import OrderedDict
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

import torch
import numpy as np
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedType
from diffusers.models import AutoencoderKL
from transformers import T5EncoderModel, T5Tokenizer
from mmcv.runner import LogBuffer
from PIL import Image
from torch.utils.data import RandomSampler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn import Conv2d
from torch.nn.parameter import Parameter

from diffusion import IDDPM, DPMS
from diffusion.data.builder import build_dataset, build_dataloader, set_data_root
from diffusion.model.builder import build_model
from diffusion.utils.checkpoint import save_checkpoint, load_checkpoint
from diffusion.utils.data_sampler import AspectRatioBatchSampler
from diffusion.utils.dist_utils import synchronize, get_world_size, clip_grad_norm_, flush, get_rank
from diffusion.utils.logger import get_root_logger, rename_file_with_creation_time
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.misc import set_random_seed, read_config, init_random_seed, DebugUnderflowOverflow
from diffusion.utils.optimizer import build_optimizer, auto_scale_lr
from diffusion.model.nets.resampler import Resampler
from src.dataset import BaseDepthDataset, DatasetMode, get_dataset
from src.dataset.depth_transform import get_depth_normalizer
from src.dataset.dist_mixed_sampler import DistributedMixedBatchSampler
from src.dataset.mixed_sampler import MixedBatchSampler
from src.dataset.hypersim_dataset import HypersimDataset
from src.utils.image_utils import decode_depth, colorize_depth_maps, chw2hwc, encode_depth
from src.utils.embedding_utils import load_null_caption_embeddings, save_null_caption_embeddings
from src.utils.multi_res_noise import multi_res_noise_like

warnings.filterwarnings("ignore")  # ignore warning


def set_fsdp_env():
    os.environ["ACCELERATE_USE_FSDP"] = 'true'
    os.environ["FSDP_AUTO_WRAP_POLICY"] = 'TRANSFORMER_BASED_WRAP'
    os.environ["FSDP_BACKWARD_PREFETCH"] = 'BACKWARD_PRE'
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = 'PixArtBlock'

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Update the EMA model parameters using vectorized operations.
    """
    with torch.no_grad():
        for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
            ema_param.mul_(decay).add_(model_param, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def _replace_patchembed_proj(model):
    """Replace the first layer to accept 8 in_channels."""
    _weight = model.x_embedder.proj.weight.clone()
    _bias = model.x_embedder.proj.bias.clone()
    _weight = _weight.repeat((1, 2, 1, 1))
    _weight *= 0.5
    _n_proj_out_channel = model.x_embedder.proj.out_channels
    kernel_size = model.x_embedder.proj.kernel_size
    padding = model.x_embedder.proj.padding
    stride = model.x_embedder.proj.stride
    _new_proj = Conv2d(
        8, _n_proj_out_channel, kernel_size=kernel_size, stride=stride, padding=padding
    )
    _new_proj.weight = Parameter(_weight)
    _new_proj.bias = Parameter(_bias)
    model.x_embedder.proj = _new_proj
    print("PatchEmbed projection layer has been replaced.")
    return model

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
    
def cosine_annealing_lr_schedule(current_step, start_step, total_steps):
    if current_step < start_step:
        return 1.0  # Keep the learning rate at the base_lr
    elif current_step >= total_steps:
        return 0.0  # Fully decay to zero after total_steps
    else:
        # Cosine annealing for the learning rate
        decay_steps = total_steps - start_step
        progress = (current_step - start_step) / decay_steps
        return 0.5 * (1 + math.cos(math.pi * progress))  # Cosine annealing


@torch.inference_mode()
def log_validation(model, loader, vae, device, step):
    torch.cuda.empty_cache()
    model = accelerator.unwrap_model(model).eval()

    print("Validation started")

    # Get the next batch
    batch = next(iter(loader))
    rgb = batch["rgb_norm"].to(device).to(train_dtype)
    rgb_int = batch["rgb_int"].to(device).to(train_dtype)  # Real RGB images from batch
    with torch.no_grad():
        bs = rgb.shape[0]
        rgb_embedding = dinvov2_model(preprocess_tensor(rgb)).unsqueeze(1)

    # Mentioning params
    batch_size = rgb.shape[0]

    wandb_images = []  # Collect all images to log at once

    for i in range(batch_size):
        with torch.no_grad():  # Map input images to latent space + normalize latents:
            rgb_input_latent = (
                vae.encode(rgb[i].unsqueeze(0)).latent_dist.sample() * vae.config.scaling_factor
            )
        
        latent_size_h, latent_size_w = rgb_input_latent.shape[2], rgb_input_latent.shape[3]
        # print(f"Latent: {rgb_input_latent.device}")
        # print(f" Device: {vae.device}")
        # print(f"Device: {next(model.parameters()).device}")
        y = resampler(rgb_embedding[i].unsqueeze(0)).unsqueeze(1)
        # y = 2 * (y - y.min()) / (y.max() - y.min()) - 1
        # emb_masks = null_caption_token.attention_mask
        # caption_embs = null_caption_embs
        null_y = torch.zeros_like(y)
        # null_y = y.repeat(1, 1, 1, 1)

        print(f'Finished embedding for image {i + 1}/{batch_size}')

        model_kwargs = {
            'data_info': None,
            'mask': None,
            'input_latent': rgb_input_latent
        }
        z = torch.randn(1, 4, latent_size_h, latent_size_w, device=device)

        # Initialize DPM-Solver
        dpm_solver = DPMS(
            model.forward_with_dpmsolver,
            condition=y,
            uncondition=null_y,
            cfg_scale=3.0,
            model_kwargs=model_kwargs
        )

        # Generate samples
        samples = dpm_solver.sample(
            z,
            steps=25,
            order=2,
            skip_type="time_uniform",
            method="multistep",
        )
        samples = samples.to(vae.dtype)
        # Decode the depth from latent space
        depth = decode_depth(samples, vae)
        depth = torch.clip(depth, -1.0, 1.0)  # TODO: Check this step
        # print(depth)
        if torch.isnan(samples).any():
            print("Samples contain NaN values")
        if torch.isnan(depth).any():
            print("Depth contains NaN values")

        # Normalize depth values between 0 and 1
        depth_pred = (depth + 1.0) / 2.0
        depth_pred = depth_pred.squeeze().detach().cpu().numpy()
        depth_pred = depth_pred.clip(0, 1)

        # Colorize depth maps using a colormap
        depth_colored = colorize_depth_maps(depth_pred, 0, 1, cmap="Spectral").squeeze()

        # Convert to uint8 for wandb logging
        depth_colored = (depth_colored * 255).astype(np.uint8)
        depth_colored_hwc = chw2hwc(depth_colored)

        # Log depth image to wandb
        wandb_images.append(wandb.Image(depth_colored_hwc, caption=f"Depth Image {i}"))

        # Also log real image from rgb_int
        real_image_np = rgb_int[i].detach().cpu().numpy()
        real_image_hwc = chw2hwc(real_image_np)
        wandb_images.append(wandb.Image(real_image_hwc, caption=f"Real Image {i}"))
        torch.cuda.empty_cache() 
        del z, rgb_input_latent, samples, depth, depth_pred, depth_colored, real_image_np


    # Log all images to wandb
    wandb.log({f"validation_images_step": wandb_images, "step": step})

    print("Validation completed and images logged to wandb.")
    gc.collect()
    torch.cuda.empty_cache()


def create_datasets(cfg, rank, world_size):
    loader_seed = 0
    num_workers = cfg.num_workers
    batch_size = cfg.train_batch_size
    val_batch_size = cfg.val_batch_size
    cfg = cfg.conf_data

    if loader_seed is None:
        loader_generator = None
    else:
        loader_generator = torch.Generator().manual_seed(loader_seed)

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

        sampler = MixedBatchSampler(
            src_dataset_ls=dataset_ls,
            batch_size=batch_size,
            drop_last=True,
            prob=cfg.dataset.train.prob_ls,
            shuffle=True,
            generator=loader_generator,
        )

        train_loader = DataLoader(
            concat_dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        sampler = DistributedSampler(
            train_dataset, 
            num_replicas=world_size, 
            rank=rank, 
            shuffle=True, 
            seed=loader_seed
        )

        train_loader = DataLoader(
            dataset=train_dataset,
            sampler=sampler,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=True,
            generator=loader_generator,
            pin_memory=True
        )

    val_dataset = HypersimDataset(
        mode=DatasetMode.TRAIN,  # Since TRAIN changes the shape
        filename_ls_path=cfg.dataset.val.filenames,
        dataset_dir=os.path.join(cfg.paths.base_data_dir, cfg.dataset.val.dir),
        resize_to_hw=cfg.dataset.val.resize_to_hw if 'resize_to_hw' in cfg.dataset.val else None,
        disp_name=cfg.dataset.val.name,
        depth_transform=depth_transform
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.validation.batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    return train_loader, val_loader

# def preprocess_tensor(tensor):
#     # Resize to 518 while maintaining aspect ratio
#     resized = torch.stack([F.resize(img, 518) for img in tensor])
#     # Center crop to [518, 518]
#     cropped = torch.stack([F.center_crop(img, (518, 518)) for img in resized])
#     # Normalize to mean=[0.5, 0.5, 0.5] and std=[0.5, 0.5, 0.5]
#     normalized = torch.stack([
#         F.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) for img in cropped
#     ])
#     return normalized
def preprocess_tensor(tensor):
    resize_transform = T.Resize((518, 518))
    resized = torch.stack([resize_transform(img) for img in tensor])
    # cropped = torch.stack([F.center_crop(img, (518, 518)) for img in resized])
    return resized

def train():
    if config.get('debug_nan', False):
        DebugUnderflowOverflow(model)
        logger.info('NaN debugger registered. Start detecting overflow during training.')

    if accelerator.sync_gradients:
        update_ema(ema, model, decay=0)  # Initialize EMA
        ema.eval()

    time_start, last_tic = time.time(), time.time()
    log_buffer = LogBuffer()
    
    global_step = start_step + 1

    epoch = 0

    # Initialize tracking (e.g., WandB)
    if accelerator.is_main_process and args.report_to == "wandb":
        wandb.init(project=args.tracker_project_name, config=config)
        accelerator.init_trackers(args.tracker_project_name, config)

    data_time_start = time.time()
    data_time_all = 0
    # Iteration-based training loop
    for step in tqdm(range(global_step, total_iterations + 1), initial=global_step, total=total_iterations, desc="Training Progress"):
        grad_norm = None
        if step % len(train_dataloader) == 0 or step==1:
            epoch += 1
            train_loader_iter = iter(train_dataloader)
        
        batch = next(train_loader_iter)  # Sample the next batch
        # logger.info(f'Step: {step}, Batch Images: {", ".join(batch["rgb_relative_path"])}')
        rgb = batch["rgb_norm"].to(device=accelerator.device, dtype=train_dtype)
        depth_gt_for_latent = batch['depth_raw_norm'].to(device=accelerator.device, dtype=train_dtype)
        bs = rgb.shape[0]
            
        if config.valid_mask_loss:
            valid_mask_for_latent = batch['valid_mask_raw']
            invalid_mask = ~valid_mask_for_latent
            valid_mask_down = ~torch.max_pool2d(invalid_mask.float(), 8, 8).bool().repeat((1, 4, 1, 1))
            valid_mask_down = valid_mask_down.to(device=accelerator.device)
        else:    
            valid_mask_down = None

        # Encode inputs
        rgb_input_latent = vae.encode(rgb).latent_dist.sample().mul_(config.scale_factor)
        depth_gt_latent = encode_depth(depth_gt_for_latent, vae)

        torch.cuda.empty_cache()

        timesteps = torch.randint(0, config.train_sampling_steps, (bs,), device=accelerator.device).long()

        data_time_all += time.time() - data_time_start

        if config.multi_res_noise is not None:
            strength = config.multi_res_noise_strength
            if config.multi_res_noise_annealing:
                # Calculate strength depending on t
                strength = strength * (timesteps / train_diffusion.num_timesteps)
                noise = multi_res_noise_like(
                    depth_gt_latent,
                    strength=strength,
                    downscale_strategy=config.multi_res_noise_downscale_strategy,
                    device=accelerator.device,
                )
        else:
            noise = None
        
        # Run inference
        with torch.no_grad():
            rgb_embedding = dinvov2_model(preprocess_tensor(rgb)).unsqueeze(1)
        y = resampler(rgb_embedding).unsqueeze(1)
        del rgb, depth_gt_for_latent, valid_mask_for_latent

        # Training step
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            loss_term = train_diffusion.training_losses(
                model, 
                depth_gt_latent, 
                timesteps, 
                model_kwargs=dict(
                    y=y, 
                    mask=None,
                    input_latent=rgb_input_latent
                ),
                valid_mask=valid_mask_down,
                noise=noise
            )
            loss = loss_term['loss'].mean()
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
                update_ema(ema, model)
                lr_scheduler.step()
        
        # Logging
        
        lr = lr_scheduler.get_last_lr()[0]
        logs = {}
        logs = {args.loss_report_name: accelerator.gather(loss).mean().item()}
        log_buffer.update(logs)

        if step % config.log_interval == 0:
            avg_time = (time.time() - time_start) / step
            eta = str(datetime.timedelta(seconds=int(avg_time * (total_iterations - step))))
            t_d = data_time_all / config.log_interval

            info = f"Step [{step}/{total_iterations}] ETA: {eta}, " \
                   f"Time Data: {t_d:.3f}, LR: {lr:.4e}, "
            info += ', '.join([f"{k}:{v:.4f}" for k, v in log_buffer.output.items()])
            logger.info(info)
            gradient_norms = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm_internal = param.grad.data.norm(2).item()
                    # Store the gradient norm with a prefix for WandB grouping
                    gradient_norms[f'gradient_norm/{name}'] = grad_norm_internal
                    
            perciever_resampler_gradients = {}
            for name, param in resampler.named_parameters():
                if param.grad is not None:
                    grad_norm_internal = param.grad.data.norm(2).item()
                    perciever_resampler_gradients[f'perciever_resampler_gradients/{name}'] = grad_norm_internal

            logs.update(gradient_norms)
            logs.update(perciever_resampler_gradients)
            logs.update(lr=lr)
            last_tic = time.time()
            log_buffer.clear()
            data_time_all = 0
            
        if grad_norm is not None:
            logs.update(grad_norm=accelerator.gather(grad_norm).mean().item())

        # Log to tracking tool (e.g., WandB)
        if accelerator.is_main_process:
            # logs.update(lr=lr)
            accelerator.log(logs, step=step)


        # Save checkpoint periodically
        if config.save_model_steps and step % config.save_model_steps == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                save_checkpoint(
                    os.path.join(config.work_dir, 'checkpoints'),
                    epoch=step // len(train_dataloader),  # Optional: calculate current epoch
                    step=step,
                    model=accelerator.unwrap_model(model),
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler
                )

        # Validation and visualization (optional)
        if config.visualize and step % config.eval_sampling_steps == 0 or step==1:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                log_validation(model, val_loader, vae, accelerator.device, step)

        # Final checkpoint after all iterations
        if step == total_iterations:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                save_checkpoint(
                    os.path.join(config.work_dir, 'checkpoints'),
                    epoch=config.num_epochs,
                    step=step,
                    model=accelerator.unwrap_model(model),
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler
                )

        # Prepare for next iteration
        global_step += 1
        data_time_start = time.time()
        # del rgb_input_latent, depth_gt_latent, timesteps
        flush()


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("config", type=str, help="config")
    parser.add_argument("--cloud", action='store_true', default=False, help="cloud or local machine")
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the dir to resume the training')
    parser.add_argument('--load-from', default=None, help='the dir to load a ckpt for training')
    parser.add_argument('--local-rank', type=int, default=-1)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--is_depth', action='store_true')
    parser.add_argument(
        "--pipeline_load_from", default='output/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers',
        type=str, help="Download for loading text_encoder, "
                       "tokenizer and vae from https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-pixart",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument("--loss_report_name", type=str, default="loss")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = read_config(args.config)
    if args.work_dir is not None:
        config.work_dir = args.work_dir
    if args.resume_from is not None:
        config.load_from = None
        config.resume_from = dict(
            checkpoint=args.resume_from,
            load_ema=False,
            resume_optimizer=True,
            resume_lr_scheduler=True)
    if args.debug:
        config.log_interval = 1
        config.train_batch_size = 1

    os.umask(0o000)
    os.makedirs(config.work_dir, exist_ok=True)

    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400)  # change timeout to avoid a strange NCCL bug
    # Initialize accelerator and tensorboard logging
    if config.use_fsdp:
        init_train = 'FSDP'
        from accelerate import FullyShardedDataParallelPlugin
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
        set_fsdp_env()
        fsdp_plugin = FullyShardedDataParallelPlugin(state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),)
    else:
        init_train = 'DDP'
        fsdp_plugin = None

    even_batches = True
    if config.multi_scale:
        even_batches=False,

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=args.report_to,
        project_dir=os.path.join(config.work_dir, "logs"),
        fsdp_plugin=fsdp_plugin,
        even_batches=even_batches,
        kwargs_handlers=[init_handler]
    )
    
    # Need to be fixed for the all other options 
    if config.mixed_precision == 'no':
        train_dtype = torch.float32
    if config.mixed_precision == 'fp16':
        train_dtype = torch.float16

    log_name = 'train_log.log'
    if accelerator.is_main_process:
        if os.path.exists(os.path.join(config.work_dir, log_name)):
            rename_file_with_creation_time(os.path.join(config.work_dir, log_name))
    logger = get_root_logger(os.path.join(config.work_dir, log_name))

    logger.info(accelerator.state)
    config.seed = init_random_seed(config.get('seed', None))
    set_random_seed(config.seed)

    world_size = get_world_size()
    rank = get_rank()

    logger.info(f"World_size: {get_world_size()}, seed: {config.seed}")
    logger.info(f"Initializing: {init_train} for training")
    image_size = config.image_size  # @param [256, 512]
    latent_size = int(image_size) // 8
    validation_noise = torch.randn(1, 4, latent_size, latent_size, device='cpu') if getattr(config, 'deterministic_validation', False) else None
    pred_sigma = getattr(config, 'pred_sigma', True)
    learn_sigma = getattr(config, 'learn_sigma', True) and pred_sigma
    max_length = config.model_max_length
    kv_compress_config = config.kv_compress_config if config.kv_compress else None
    vae = None
    vae = AutoencoderKL.from_pretrained(config.vae_pretrained, torch_dtype=train_dtype).to(accelerator.device)
    config.scale_factor = vae.config.scaling_factor

    logger.info(f"vae scale factor: {config.scale_factor}")

    max_length = 16
    model_kwargs = {"pe_interpolation": config.pe_interpolation, "config": config,
                    "model_max_length": max_length, "qk_norm": config.qk_norm,
                    "kv_compress_config": kv_compress_config, "micro_condition": config.micro_condition}
    

    # Check if the .pt files exist, otherwise save them
    save_dir = f"output/null_embedding/{max_length}"
    if not (os.path.exists(os.path.join(save_dir, "null_caption_token.pt")) and
            os.path.exists(os.path.join(save_dir, "null_caption_embs.pt"))):
        save_null_caption_embeddings(args.pipeline_load_from, max_length, accelerator.device,save_dir)

    # Load the saved embeddings and tokens
    null_caption_token, null_caption_embs = load_null_caption_embeddings(save_dir)
    null_caption_embs = null_caption_embs.to(accelerator.device)
    null_caption_token = null_caption_token.to(accelerator.device)

    # build models
    train_diffusion = IDDPM(str(config.train_sampling_steps), learn_sigma=learn_sigma, pred_sigma=pred_sigma, snr=config.snr_loss)
    model = build_model(config.model,
                        config.grad_checkpointing,
                        config.get('fp32_attention', False),
                        input_size=latent_size,
                        learn_sigma=learn_sigma,
                        pred_sigma=pred_sigma,
                        **model_kwargs).train()
    logger.info(f"{model.__class__.__name__} Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    if args.load_from is not None:
        config.load_from = args.load_from

    def load_model():
        if config.load_from is not None:
            missing, unexpected = load_checkpoint(
                config.load_from, model, load_ema=config.get('load_ema', False), max_length=max_length)
            logger.warning(f'Missing keys: {missing}')
            logger.warning(f'Unexpected keys: {unexpected}')


    if args.is_depth:
        # For Depth DiT model
        # The state_dict is already modified for depth, so modify the model before loading
        model = _replace_patchembed_proj(model)
        load_model()
    else:
        # For a vanilla DiT model
        # Load the state_dict first, and then modify the model afterwards
        load_model()
        model = _replace_patchembed_proj(model)
    
    ema = deepcopy(model).to(accelerator.device)  # EMA model
    requires_grad(ema, False)
    
    # prepare for FSDP clip grad norm calculation
    if accelerator.distributed_type == DistributedType.FSDP:
        for m in accelerator._models:
            m.clip_grad_norm_ = types.MethodType(clip_grad_norm_, m)

    dinvov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
    dinvov2_model = accelerator.prepare(dinvov2_model)
    model.eval()

    # build dataloader
    train_dataloader, val_loader = create_datasets(config, rank, world_size)
    logger.info(f"Number of training samples: {len(train_dataloader.dataset)}")
    logger.info(f"Total number of batches: {len(train_dataloader)}")
    logger.info(f"Batch size per GPU: {config.train_batch_size}")
    logger.info(f"Num Workers: {config.num_workers}")
    logger.info(f"Effective Batch size: {config.train_batch_size * world_size * config.gradient_accumulation_steps}")
    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    resampler = Resampler(num_queries=max_length).train()
    resampler.requires_grad_(True)

    # optimizer
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(resampler.parameters()), lr=config.lr, weight_decay=config.weight_decay)

    if hasattr(config, 'num_iterations'):
        total_iterations = config.num_iterations
    else:
        total_iterations = config.num_epochs * len(train_dataloader)  # Calculate total iterations

    if config.lr_scheduler:
        if config.cosine_annealing:
            lr_scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing_lr_schedule(step, start_step=config.start_step, total_steps=total_iterations),
            )

        else:
            warmup = config.warmup_steps if hasattr(config, 'warmup_steps') else 0
            stop_step = config.stop_step if hasattr(config, 'stop_step') else total_iterations
            final_lr = config.final_lr if hasattr(config, 'final_lr') else 1e-10 # some very small value
                
            lr_scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: linear_decay_lr_schedule(step, warmup_steps=warmup, start_step=config.start_step, stop_step=stop_step, base_lr=config.lr, final_lr=final_lr),
            )

    if accelerator.is_main_process:
        tracker_config = dict(vars(config))
        try:
            accelerator.init_trackers(args.tracker_project_name, tracker_config)
        except:
            accelerator.init_trackers(f"tb_{timestamp}")

    start_epoch = 0
    start_step = 0
    skip_step = config.skip_step
    total_steps = len(train_dataloader) * config.num_epochs

    if config.resume_from is not None and config.resume_from['checkpoint'] is not None:
        resume_path = config.resume_from['checkpoint']
        path = os.path.basename(resume_path)
        start_epoch = int(path.replace('.pth', '').split("_")[1]) - 1
        start_step = int(path.replace('.pth', '').split("_")[3])
        _, missing, unexpected = load_checkpoint(**config.resume_from,
                                                 model=model,
                                                 optimizer=optimizer,
                                                 lr_scheduler=lr_scheduler,
                                                 max_length=max_length,
                                                 )

        logger.warning(f'Missing keys: {missing}')
        logger.warning(f'Unexpected keys: {unexpected}')
    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    # model, resampler = accelerator.prepare(model, resampler)
    model, resampler, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, resampler, optimizer, train_dataloader, lr_scheduler)
    train()

"""
python -m torch.distributed.launch --nproc_per_node=2 --master_port=12345 \
          train_scripts/train_depth.py \
          configs/pixart_sigma_config/PixArt_sigma_xl2_img512_depth.py \
          --load-from /mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/depth_estimation_experiments/PixArt-sigma/output/pretrained_models/PixArt-Sigma-XL-2-512-MS.pth \
          --work-dir output/depth_512_mixed_training \
          --debug \
          --report_to wandb
"""

"""
python -m torch.distributed.launch --nproc_per_node=2 --master_port=12345 \
          train_scripts/train_depth.py \
          configs/pixart_sigma_config/PixArt_sigma_xl2_img512_depth.py \
          --load-from /mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/depth_estimation_experiments/DiT/PixArt-sigma/output/depth_512_mixed_training_iter3/checkpoints/epoch_5_step_22000.pth \
          --work-dir output/depth_512_mixed_training \
          --is_depth \
          --debug \
          --report_to wandb
"""

#TODO
"""
- log epoch and steps in ever steps
- log batch files in offline training logger
"""

'''
import numpy as np
from collections import deque

# Parameters
window_size = 50  # Size of the sliding window

# Initialize a deque to store the most recent losses
loss_window = deque(maxlen=window_size)

def calculate_loss_variance(loss):
    """
    Updates the sliding window with the new loss value and calculates variance.
    
    Args:
        loss (float): Current loss value from training.

    Returns:
        float: Variance of the losses in the sliding window. Returns None if window is not full yet.
    """
    # Add the current loss to the sliding window
    loss_window.append(loss)
    
    # Only calculate variance if the window is full
    if len(loss_window) == window_size:
        # Convert deque to numpy array for computation
        loss_array = np.array(loss_window)
        # Calculate mean
        mean_loss = np.mean(loss_array)
        # Calculate variance
        variance = np.mean((loss_array - mean_loss) ** 2)
        return variance
    else:
        # Return None if the window is not full
        return None

# Example usage in training loop
for iteration in range(1, 1000):  # Simulate 1000 iterations
    # Simulated loss value (replace this with actual loss)
    simulated_loss = np.random.random() * 0.1 + 0.02  # Simulated small random loss
    
    # Calculate loss variance
    variance = calculate_loss_variance(simulated_loss)
    
    if variance is not None:
        print(f"Iteration {iteration}: Loss Variance = {variance:.6f}")

'''