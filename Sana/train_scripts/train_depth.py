# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
import autoroot
import autorootcwd
import datetime
import getpass
import hashlib
import json
import os

import os.path as osp
import random
import time
import types
import warnings
from dataclasses import asdict
from pathlib import Path
import sys
sys.path.append(os.path.abspath(".."))

import numpy as np
import pyrallis
import torch
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedType
from PIL import Image
from termcolor import colored
import wandb
warnings.filterwarnings("ignore")  # ignore warning

from src.dataset import BaseDepthDataset, DatasetMode, get_dataset
from src.dataset.depth_transform import get_depth_normalizer
from src.dataset.dist_mixed_sampler import DistributedMixedBatchSampler
from src.dataset.mixed_sampler import MixedBatchSampler
from src.dataset.hypersim_dataset import HypersimDataset
from src.utils.image_utils import colorize_depth_maps, chw2hwc
from src.utils.embedding_utils import load_null_caption_embeddings, save_null_caption_embeddings
from src.utils.multi_res_noise import multi_res_noise_like


from diffusion import DPMS, FlowEuler, Scheduler
from diffusion.data.builder import build_dataloader, build_dataset
from diffusion.data.wids import DistributedRangedSampler
from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder, get_vae, vae_decode, vae_encode
from diffusion.model.respace import compute_density_for_timestep_sampling
from diffusion.utils.checkpoint import load_checkpoint, save_checkpoint
from diffusion.utils.config import SanaConfig
from diffusion.utils.data_sampler import AspectRatioBatchSampler
from diffusion.utils.dist_utils import clip_grad_norm_, flush, get_world_size
from diffusion.utils.logger import LogBuffer, get_root_logger
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.misc import DebugUnderflowOverflow, init_random_seed, read_config, set_random_seed
from diffusion.utils.optimizer import auto_scale_lr, build_optimizer
from diffusion.data.datasets.utils import ASPECT_RATIO_256, ASPECT_RATIO_512, ASPECT_RATIO_1024, ASPECT_RATIO_2048, ASPECT_RATIO_2880, ASPECT_RATIO_4096

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_fsdp_env():
    os.environ["ACCELERATE_USE_FSDP"] = "true"
    os.environ["FSDP_AUTO_WRAP_POLICY"] = "TRANSFORMER_BASED_WRAP"
    os.environ["FSDP_BACKWARD_PREFETCH"] = "BACKWARD_PRE"
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = "SanaBlock"

def convert_depth_to_colored(depth):
        # Normalize depth values between 0 and 1
        depth_pred = (depth + 1.0) / 2.0
        depth_pred = depth_pred.squeeze().detach().cpu().numpy()
        depth_pred = depth_pred.clip(0, 1)

        # Colorize depth maps using a colormap
        depth_colored = colorize_depth_maps(depth_pred, 0, 1, cmap="Spectral").squeeze()

        # Convert to uint8 for wandb logging
        depth_colored = (depth_colored * 255).astype(np.uint8)
        depth_colored_hwc = chw2hwc(depth_colored)
        return depth_colored_hwc


@torch.inference_mode()
def log_validation(accelerator, config, model, logger, step, device, vae=None, init_noise=None):
    torch.cuda.empty_cache()
    model = accelerator.unwrap_model(model).eval()
    vis_sampler = config.scheduler.vis_sampler
    wandb_images = []  # Collect all images to log at once
    for task in ['depth_pred', 'rgb_pred']:
        print(f"Validation started for {task}")
        for idx, batch in enumerate(val_dataloader):
            rgb = batch["rgb_norm"].to(device).to(torch.float16)
            rgb_int = batch["rgb_int"].to(device).to(torch.float16)  # Real RGB images from batch
            gt_depth = batch["depth_raw_norm"].to(device).to(torch.float16) # GT Depth images from batch
            
            # Map input image to latent space + normalize latents
            with torch.no_grad():
                input_latent = vae_encode(config.vae.vae_type, vae, rgb, config.vae.sample_posterior, accelerator.device)
            latent_size_h, latent_size_w = input_latent.shape[2], input_latent.shape[3]
            z = torch.randn(1, config.vae.vae_latent_dim, latent_size_h, latent_size_w, device=device)
            
            # task=depth and batch_size=1
            if task =='depth_pred':
                task_emb = torch.tensor([1, 0]).float().unsqueeze(0).repeat(1, 1).to(accelerator.device)
                task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1).repeat(1, 1)
            elif task == 'rgb_pred':
                task_emb = torch.tensor([0, 1]).float().unsqueeze(0).repeat(1, 1).to(accelerator.device)
                task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1).repeat(1, 1)
            else:
                raise ValueError(f"Unknown task: {task}")
            # Embedding preparation
            emb_masks = null_caption_token.attention_mask
            caption_embs = null_caption_embs
            null_y = null_caption_embs.repeat(1, 1, 1)
            print(f"Finished embedding for image {idx + 1}/{len(val_dataloader)}")
            # if input_latent is not None:
            input_latent = torch.cat([input_latent] * 2)

            model_kwargs = dict(
                data_info=None,
                mask=emb_masks,
                input_latent=input_latent,
                task_emb=task_emb
            )
            
            if vis_sampler == "dpm-solver":
                dpm_solver = DPMS(
                    model.forward_with_dpmsolver,
                    condition=caption_embs,
                    uncondition=null_y,
                    cfg_scale=4.5,
                    model_kwargs=model_kwargs,
                )
                denoised = dpm_solver.sample(
                    z,
                    steps=14,
                    order=2,
                    skip_type="time_uniform",
                    method="multistep",
                )
            elif vis_sampler == "flow_euler":
                flow_solver = FlowEuler(
                    model, condition=caption_embs, uncondition=null_y, cfg_scale=4.5, model_kwargs=model_kwargs
                )
                denoised = flow_solver.sample(z, steps=28)
            elif vis_sampler == "flow_dpm-solver":
                dpm_solver = DPMS(
                    model.forward_with_dpmsolver,
                    condition=caption_embs,
                    uncondition=null_y,
                    cfg_scale=4.5,
                    model_type="flow",
                    model_kwargs=model_kwargs,
                    schedule="FLOW",
                )
                denoised = dpm_solver.sample(
                    z,
                    steps=20,
                    order=2,
                    skip_type="time_uniform_flow",
                    method="multistep",
                    flow_shift=config.scheduler.flow_shift,
                )
            else:
                raise ValueError(f"{vis_sampler} not implemented")

            latent = denoised.to(next(vae.parameters()).dtype)
            
            if task == 'depth_pred':
                # Decode the depth from latent space
                depth = decode_depth(config.vae.vae_type, vae, latent)
                depth = torch.clip(depth, -1.0, 1.0)  

                # calulate colored depth
                depth_colored_hwc = convert_depth_to_colored(depth)
                gt_depth_colored_hwc = convert_depth_to_colored(gt_depth.unsqueeze(0))

                # Log depth image to wandb
                wandb_images.append(wandb.Image(depth_colored_hwc, caption=f"Depth Image {idx}"))
                # Log Gt depth
                wandb_images.append(wandb.Image(gt_depth_colored_hwc, caption=f"Gt Depth {idx}"))
                
                del depth, depth_colored_hwc
            else:
                samples = vae_decode(config.vae.vae_type, vae, latent)
                samples = (
                    torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
                )
                image = Image.fromarray(samples)
                wandb_images.append(wandb.Image(image, caption=f"Reconstructed Image {idx}"))
                del samples, image

            # Also log real image from rgb_int
            real_image_np = rgb_int.detach().cpu().numpy()
            real_image_hwc = chw2hwc(real_image_np[0])            
            wandb_images.append(wandb.Image(real_image_hwc, caption=f"Real Image {idx}"))
            
            del z, input_latent, latent
            del real_image_hwc
            del null_y, caption_embs, emb_masks
            flush()
        # Log all images to wandb
        wandb.log({f"validation_images_step_{task}": wandb_images, "step": step})
        wandb_images.clear()

    print("Validation completed and images logged to wandb.")
    del vae
    flush()

def decode_depth(name, vae, depth_latent):
    """Decode depth latent into depth map."""
    if name == "sdxl" or name == "sd3":
        depth_latent = (depth_latent.detach() / vae.config.scaling_factor) + vae.config.shift_factor
        z = vae.post_quant_conv(depth_latent)
    elif "dc-ae" in name:
        # ae = vae
        z = depth_latent.detach() / vae.cfg.scaling_factor
        # z = ae.decode(depth_latent.detach() / ae.cfg.scaling_factor)
    stacked = vae.decode(z)
    depth_mean = stacked.mean(dim=1, keepdim=True)
    del stacked, z
    return depth_mean

def stack_depth_images(depth_in):
    if 4 == len(depth_in.shape):
        stacked = depth_in.repeat(1, 3, 1, 1)
    elif 3 == len(depth_in.shape):
        stacked = depth_in.unsqueeze(1)
        stacked = depth_in.repeat(1, 3, 1, 1)
    return stacked

def encode_depth(name, vae, depth_in, sample_posterior, device):
    # stack depth into 3-channel
    stacked = stack_depth_images(depth_in)
    # encode using VAE encoder
    depth_latent = vae_encode(name, vae, stacked, sample_posterior, device)
    del stacked, depth_in
    return depth_latent

def train(config, args, accelerator, model, optimizer, lr_scheduler, train_dataloader, train_diffusion, logger):
    if getattr(config.train, "debug_nan", False):
        DebugUnderflowOverflow(model)
        logger.info("NaN debugger registered. Start to detect overflow during training.")
    log_buffer = LogBuffer()

    global_step = start_step + 1
    skip_step = max(config.train.skip_step, global_step) % train_dataloader_len
    skip_step = skip_step if skip_step < (train_dataloader_len - 20) else 0
    loss_nan_timer = 0

    # Now you train the model
    for epoch in range(start_epoch + 1, config.train.num_epochs + 1):
        time_start, last_tic = time.time(), time.time()
        sampler = train_dataloader.batch_sampler
        sampler.set_epoch(epoch) 
        # sampler.set_start(max((skip_step - 1) * config.train.train_batch_size, 0)) # skip_step is not relevant to us
        if skip_step > 1 and accelerator.is_main_process:
            logger.info(f"Skipped Steps: {skip_step}")
        skip_step = 1
        data_time_start = time.time()
        data_time_all = 0
        lm_time_all = 0
        vae_time_all = 0
        model_time_all = 0
        rgb_loss, depth_loss = float('inf'), float('inf')
        for step, batch in enumerate(train_dataloader):
            # image, json_info, key = batch
            
            rgb = batch["rgb_norm"].to(device=accelerator.device, dtype=torch.float16) # [B, 3, H, W]
            depth_gt_for_latent = batch['depth_raw_norm'].to(device=accelerator.device, dtype=torch.float16) # [B, 1, H, W]
            bs = rgb.shape[0]

            accelerator.wait_for_everyone()
            data_time_all += time.time() - data_time_start
            vae_time_start = time.time()
            rgb_input_latent = vae_encode(config.vae.vae_type, vae, rgb, config.vae.sample_posterior, accelerator.device)
            gt_latent = encode_depth(config.vae.vae_type, vae, depth_gt_for_latent, config.vae.sample_posterior, accelerator.device)
            # Randomly decide the task
            task = 'depth_pred' if torch.rand(1).item() < 0.5 else 'rgb_pred'
            # Set gt_latent and task_emb based on the chosen task
            if task == 'depth_pred':
                gt_latent = encode_depth(config.vae.vae_type, vae, depth_gt_for_latent, config.vae.sample_posterior, accelerator.device)
                task_emb_values = [1, 0]
            elif task == 'rgb_pred':
                gt_latent = rgb_input_latent
                task_emb_values = [0, 1]
            else:
                raise ValueError(f"Unknown task: {task}")
            
            task_emb = torch.tensor(task_emb_values).float().unsqueeze(0).to(accelerator.device)
            task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1).repeat(bs, 1)

            del rgb, depth_gt_for_latent

            if getattr(config, "valid_mask_loss", False):  # Default to False if valid_mask_loss doesn't exist
                valid_mask_for_latent = batch['valid_mask_raw']
                invalid_mask = ~valid_mask_for_latent
                # vae.spatial_compression_ratio: 32 # vae.cfg.latent_channels: 32
                valid_mask_down = ~torch.max_pool2d(
                    invalid_mask.float(), 
                    vae.spatial_compression_ratio, 
                    vae.spatial_compression_ratio
                ).bool().repeat((1, vae.cfg.latent_channels, 1, 1))
                valid_mask_down = valid_mask_down.to(device=accelerator.device)
            else:    
                valid_mask_down = None

            accelerator.wait_for_everyone()
            vae_time_all += time.time() - vae_time_start
            lm_time_start = time.time()
            
            
            timesteps = torch.randint(
                0, config.scheduler.train_sampling_steps, (bs,), device=gt_latent.device
            ).long()
            if config.scheduler.weighting_scheme in ["logit_normal"]:
                # adapting from diffusers.training_utils
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=config.scheduler.weighting_scheme,
                    batch_size=bs,
                    logit_mean=config.scheduler.logit_mean,
                    logit_std=config.scheduler.logit_std,
                    mode_scale=None,  # not used
                )
                timesteps = (u * config.scheduler.train_sampling_steps).long().to(gt_latent.device)
            
                
            if getattr(config, "multi_res_noise", None):
                strength = getattr(config.multi_res_noise, "strength", 1.0)
                if getattr(config.multi_res_noise, "annealing", False):
                    strength = strength * (timesteps / train_diffusion.num_timesteps)
                noise = multi_res_noise_like(
                    gt_latent,
                    strength=strength,
                    downscale_strategy=getattr(config.multi_res_noise, "downscale_strategy", "original"),
                    device=accelerator.device,
                )
            else:
                noise = None

            grad_norm = None
            accelerator.wait_for_everyone()
            lm_time_all += time.time() - lm_time_start
            model_time_start = time.time()
            with accelerator.accumulate(model):
                # Predict the noise residual
                optimizer.zero_grad()
                loss_term = train_diffusion.training_losses(
                    model, 
                    gt_latent, 
                    timesteps, 
                    valid_mask=valid_mask_down,
                    noise=noise,
                    model_kwargs=dict(
                        y=null_caption_embs.repeat(bs, 1, 1, 1), #y = null_caption_embs.repeat(bs, 1, 1, 1)
                        mask=mask.repeat(bs, 1, 1, 1), #y_mask = mask.repeat(bs, 1, 1, 1) 
                        data_info=None, 
                        input_latent=rgb_input_latent,
                        task_emb=task_emb
                    )
                )
                loss = loss_term["loss"].mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.train.gradient_clip)
                optimizer.step()
                lr_scheduler.step()
                accelerator.wait_for_everyone()
                model_time_all += time.time() - model_time_start
            del gt_latent, noise, rgb_input_latent, valid_mask_down, task_emb
            if task == 'depth_pred':
                depth_loss = loss.detach().item() 
            elif task == 'rgb_pred':
                rgb_loss = loss.detach().item() 
            else:
                raise ValueError(f"Unknown task: {task}")
            if torch.any(torch.isnan(loss)):
                loss_nan_timer += 1
            lr = lr_scheduler.get_last_lr()[0]
            logs = {args.loss_report_name: accelerator.gather(loss).mean().item()}
            logs.update(depth_loss=depth_loss)
            logs.update(rgb_loss=rgb_loss)
            if grad_norm is not None:
                logs.update(grad_norm=accelerator.gather(grad_norm).mean().item())
            log_buffer.update(logs)
            if (step + 1) % config.train.log_interval == 0 or (step + 1) == 1:
                accelerator.wait_for_everyone()
                t = (time.time() - last_tic) / config.train.log_interval
                t_d = data_time_all / config.train.log_interval
                t_m = model_time_all / config.train.log_interval
                t_lm = lm_time_all / config.train.log_interval
                t_vae = vae_time_all / config.train.log_interval
                avg_time = (time.time() - time_start) / (step + 1)
                eta = str(datetime.timedelta(seconds=int(avg_time * (total_steps - global_step - 1))))
                # eta_epoch = str(
                #     datetime.timedelta(
                #         seconds=int(
                #             avg_time
                #             * (train_dataloader_len - sampler.step_start // config.train.train_batch_size - step - 1)
                #         )
                #     )
                # )
                eta_epoch = ""
                log_buffer.average()

                # current_step = (
                #     global_step - sampler.step_start // config.train.train_batch_size
                # ) % train_dataloader_len
                # current_step = train_dataloader_len if current_step == 0 else current_step
                current_step = 0
                info = (
                    f"Epoch: {epoch} | Global Step: {global_step} | Local Step: {current_step} // {train_dataloader_len}, "
                    f"total_eta: {eta}, epoch_eta:{eta_epoch}, time: all:{t:.3f}, model:{t_m:.3f}, data:{t_d:.3f}, "
                    f"lm:{t_lm:.3f}, vae:{t_vae:.3f}, lr:{lr:.3e}, "
                )
                info += (
                    f"s:({model.module.h}, {model.module.w}), "
                    if hasattr(model, "module")
                    else f"s:({model.h}, {model.w}), "
                )
                
                gradient_norms = {}
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm_internal = param.grad.data.norm(2).item()
                        # Store the gradient norm with a prefix for WandB grouping
                        gradient_norms[f'gradient_norm/{name}'] = grad_norm_internal
                logs.update(gradient_norms)

                info += ", ".join([f"{k}:{v:.4f}" for k, v in log_buffer.output.items()])
                last_tic = time.time()
                log_buffer.clear()
                data_time_all = 0
                model_time_all = 0
                lm_time_all = 0
                vae_time_all = 0
                if accelerator.is_main_process:
                    logger.info(info)

            logs.update(lr=lr)
            if accelerator.is_main_process:
                accelerator.log(logs, step=global_step)

            global_step += 1

            if (
                global_step % config.train.save_model_steps == 0
                or (time.time() - training_start_time) / 3600 > config.train.training_hours
            ):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    os.umask(0o000)
                    ckpt_saved_path = save_checkpoint(
                        osp.join(config.work_dir, "checkpoints"),
                        epoch=epoch,
                        step=global_step,
                        model=accelerator.unwrap_model(model),
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        generator=generator,
                        add_symlink=True,
                    )
                    if config.train.online_metric and global_step % config.train.eval_metric_step == 0 and step > 1:
                        online_metric_monitor_dir = osp.join(config.work_dir, config.train.online_metric_dir)
                        os.makedirs(online_metric_monitor_dir, exist_ok=True)
                        with open(f"{online_metric_monitor_dir}/{ckpt_saved_path.split('/')[-1]}.txt", "w") as f:
                            f.write(osp.join(config.work_dir, "config.py") + "\n")
                            f.write(ckpt_saved_path)

            if config.train.visualize and (global_step % config.train.eval_sampling_steps == 0 or global_step in [2, 50, 100, 250, 500]):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    if validation_noise is not None:
                        log_validation(
                            accelerator=accelerator,
                            config=config,
                            model=model,
                            logger=logger,
                            step=global_step,
                            device=accelerator.device,
                            vae=vae,
                            init_noise=validation_noise,
                        )
                    else:
                        log_validation(
                            accelerator=accelerator,
                            config=config,
                            model=model,
                            logger=logger,
                            step=global_step,
                            device=accelerator.device,
                            vae=vae,
                        )


            data_time_start = time.time()

        if epoch % config.train.save_model_epochs == 0 or epoch == config.train.num_epochs and not config.debug:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                # os.umask(0o000)
                ckpt_saved_path = save_checkpoint(
                    osp.join(config.work_dir, "checkpoints"),
                    epoch=epoch,
                    step=global_step,
                    model=accelerator.unwrap_model(model),
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    generator=generator,
                    add_symlink=True,
                )

                online_metric_monitor_dir = osp.join(config.work_dir, config.train.online_metric_dir)
                os.makedirs(online_metric_monitor_dir, exist_ok=True)
                with open(f"{online_metric_monitor_dir}/{ckpt_saved_path.split('/')[-1]}.txt", "w") as f:
                    f.write(osp.join(config.work_dir, "config.py") + "\n")
                    f.write(ckpt_saved_path)
        accelerator.wait_for_everyone()

def create_datasets(cfg, rank, world_size):
    if cfg.train.seed is None:
        loader_generator = None
    else:
        loader_generator = torch.Generator().manual_seed(cfg.train.seed)

    cfg.model.aspect_ratio_type == ""
    # Training dataset
    depth_transform = get_depth_normalizer(
        cfg_normalizer=cfg.depth_normalization
    )

    aspect_ratio_map = globals().get(cfg.model.aspect_ratio_type, None)
        
    train_dataset: BaseDepthDataset = get_dataset(
        cfg.dataset.train,
        base_data_dir=cfg.paths.base_data_dir,
        mode=DatasetMode.TRAIN,
        augmentation_args=cfg.augmentation,
        depth_transform=depth_transform,
        aspect_ratio_map=aspect_ratio_map
    )

    if "mixed" == cfg.dataset.train.name:
        dataset_ls = train_dataset
        assert len(cfg.dataset.train.prob_ls) == len(
            dataset_ls
        ), "Lengths don't match: `prob_ls` and `dataset_list`"
        concat_dataset = ConcatDataset(dataset_ls)

        sampler = DistributedMixedBatchSampler(
            src_dataset_ls=dataset_ls,
            batch_size=cfg.train.train_batch_size,
            drop_last=True,
            shuffle=True,
            world_size=world_size,
            rank=rank,
            prob=cfg.dataset.train.prob_ls,
            generator=loader_generator,
        )

        train_loader = DataLoader(
            concat_dataset,
            batch_sampler=sampler,
            num_workers=cfg.train.num_workers,
        )
    else:
        sampler = DistributedSampler(
            train_dataset, 
            num_replicas=world_size, 
            rank=rank, 
            shuffle=True,
            drop_last=True, 
            seed=cfg.train.seed
        )

        train_loader = DataLoader(
            dataset=train_dataset,
            sampler=sampler,
            batch_size=cfg.train.train_batch_size,
            num_workers=cfg.train.num_workers,
            drop_last=True,
            generator=loader_generator,
        )
    
    # For Validation 
    val_dataset: BaseDepthDataset = get_dataset(
            cfg.dataset.val,
            base_data_dir=cfg.paths.base_data_dir,
            mode=DatasetMode.TRAIN,
            depth_transform=depth_transform,
            drop_last=True,
            aspect_ratio_map=aspect_ratio_map
        )
    
    if "mixed" == cfg.dataset.val.name:
        val_dataset = ConcatDataset(val_dataset)

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.validation.batch_size,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=False
    )
    return train_loader, val_loader

@pyrallis.wrap()
def main(cfg: SanaConfig) -> None:
    global train_dataloader_len, start_epoch, start_step, vae, generator, num_replicas, rank, training_start_time
    global load_vae_feat, load_text_feat, validation_noise
    global max_length, validation_prompts, latent_size, valid_prompt_embed_suffix, null_embed_path
    global image_size, cache_file, total_steps
    global mask, null_caption_embs, null_caption_token, val_dataloader

    config = cfg
    args = cfg
    # config = read_config(args.config)

    training_start_time = time.time()
    load_from = True
    if args.resume_from or config.model.resume_from:
        load_from = False
        config.model.resume_from = dict(
            checkpoint=args.resume_from or config.model.resume_from,
            load_ema=False,
            resume_optimizer=True,
            resume_lr_scheduler=True,
        )
    # if resume_from_depth:
        
    if args.debug:
        config.train.log_interval = 1
        config.train.train_batch_size = min(64, config.train.train_batch_size)
        # args.report_to = "tensorboard"

    os.umask(0o000)
    os.makedirs(config.work_dir, exist_ok=True)

    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400)  # change timeout to avoid a strange NCCL bug
    # Initialize accelerator and tensorboard logging
    if config.train.use_fsdp:
        init_train = "FSDP"
        from accelerate import FullyShardedDataParallelPlugin
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig

        set_fsdp_env()
        fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
        )
    else:
        init_train = "DDP"
        fsdp_plugin = None

    accelerator = Accelerator(
        mixed_precision=config.model.mixed_precision,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        log_with=args.report_to,
        project_dir=osp.join(config.work_dir, "logs"),
        fsdp_plugin=fsdp_plugin,
        kwargs_handlers=[init_handler],
    )

    log_name = "train_log.log"
    logger = get_root_logger(osp.join(config.work_dir, log_name))
    logger.info(accelerator.state)

    config.train.seed = init_random_seed(getattr(config.train, "seed", None))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))  # Default to 0 if LOCAL_RANK is not set
    set_random_seed(config.train.seed + local_rank)
    generator = torch.Generator(device="cpu").manual_seed(config.train.seed)

    if accelerator.is_main_process:
        pyrallis.dump(config, open(osp.join(config.work_dir, "config.yaml"), "w"), sort_keys=False, indent=4)
        if args.report_to == "wandb":
            import wandb
            # wandb.init(project=args.tracker_project_name, name=args.name, resume="allow", id=args.name)
            wandb.init(project=args.tracker_project_name)
            wandb.run.log_code(".")

            

    # logger.info(f"Config: \n{config}")
    logger.info(f"World_size: {get_world_size()}, seed: {config.train.seed}")
    logger.info(f"Initializing: {init_train} for training")

    image_size = config.model.image_size
    latent_size = int(image_size) // config.vae.vae_downsample_rate
    pred_sigma = getattr(config.scheduler, "pred_sigma", True)
    learn_sigma = getattr(config.scheduler, "learn_sigma", True) and pred_sigma
    max_length = config.text_encoder.model_max_length
    vae = None
    validation_noise = (
        torch.randn(1, config.vae.vae_latent_dim, latent_size, latent_size, device="cpu", generator=generator)
        if getattr(config.train, "deterministic_validation", False)
        else None
    )
    if not config.data.load_vae_feat:
        vae = get_vae(config.vae.vae_type, config.vae.vae_pretrained, accelerator.device).to(torch.float16)
    
    tokenizer = text_encoder = None
    if not config.data.load_text_feat:
        tokenizer, text_encoder = get_tokenizer_and_text_encoder(
            name=config.text_encoder.text_encoder_name, device=accelerator.device
        )
        text_embed_dim = text_encoder.config.hidden_size
    else:
        text_embed_dim = config.text_encoder.caption_channels
    del tokenizer, text_encoder

    os.makedirs(config.train.null_embed_root, exist_ok=True)
    null_embed_path = osp.join(
        config.train.null_embed_root,
        f"null_embed_diffusers_{config.text_encoder.text_encoder_name}_{max_length}token_{text_embed_dim}.pth",
    )
    
    save_dir = f"output/null_embedding/{max_length}"
    if accelerator.is_main_process:
        # Check if the .pt files exist, otherwise save them
        # Check if embeddings and tokens already exist
        if not (os.path.exists(os.path.join(save_dir, "null_caption_token.pt")) and
                os.path.exists(os.path.join(save_dir, "null_caption_embs.pt"))):
            # Ensure directory exists
            os.makedirs(save_dir, exist_ok=True)
            save_null_caption_embeddings(
                encoder_name=cfg.text_encoder.text_encoder_name,
                max_sequence_length=max_length,
                device=accelerator.device,
                save_dir=save_dir
            )

    # Load the saved embeddings and tokens
    null_caption_token, null_caption_embs = load_null_caption_embeddings(save_dir)
    null_caption_embs = null_caption_embs.to(accelerator.device)
    null_caption_token = null_caption_token.to(accelerator.device)
    mask = null_caption_token["attention_mask"].unsqueeze(0)
    
    os.environ["AUTOCAST_LINEAR_ATTN"] = "true" if config.model.autocast_linear_attn else "false"

    # 1. build scheduler
    train_diffusion = Scheduler(
        str(config.scheduler.train_sampling_steps),
        noise_schedule=config.scheduler.noise_schedule,
        predict_v=config.scheduler.predict_v,
        learn_sigma=learn_sigma,
        pred_sigma=pred_sigma,
        snr=config.train.snr_loss,
        flow_shift=config.scheduler.flow_shift,
    )
    predict_info = f"v-prediction: {config.scheduler.predict_v}, noise schedule: {config.scheduler.noise_schedule}"
    if "flow" in config.scheduler.noise_schedule:
        predict_info += f", flow shift: {config.scheduler.flow_shift}"
    if config.scheduler.weighting_scheme in ["logit_normal", "mode"]:
        predict_info += (
            f", flow weighting: {config.scheduler.weighting_scheme}, "
            f"logit-mean: {config.scheduler.logit_mean}, logit-std: {config.scheduler.logit_std}"
        )
    logger.info(predict_info)

    # 2. build models
    model_kwargs = {
        "pe_interpolation": config.model.pe_interpolation,
        "config": config,
        "model_max_length": max_length,
        "qk_norm": config.model.qk_norm,
        "micro_condition": config.model.micro_condition,
        "caption_channels": text_embed_dim,
        "y_norm": config.text_encoder.y_norm,
        "attn_type": config.model.attn_type,
        "ffn_type": config.model.ffn_type,
        "mlp_ratio": config.model.mlp_ratio,
        "mlp_acts": list(config.model.mlp_acts),
        "in_channels": config.vae.vae_latent_dim,
        "y_norm_scale_factor": config.text_encoder.y_norm_scale_factor,
        "use_pe": config.model.use_pe,
        "linear_head_dim": config.model.linear_head_dim,
        "pred_sigma": pred_sigma,
        "learn_sigma": learn_sigma,
        "depth_channels":32
    }
    model = build_model(
        config.model.model,
        config.train.grad_checkpointing,
        getattr(config.model, "fp32_attention", False),
        input_size=latent_size,
        **model_kwargs,
    ).train()
    logger.info(
        colored(
            f"{model.__class__.__name__}:{config.model.model}, "
            f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M",
            "green",
            attrs=["bold"],
        )
    )
    # 2-1. load model
    if args.load_from is not None:
        config.model.load_from = args.load_from
    if config.model.load_from is not None and load_from:
        _, missing, unexpected, _ = load_checkpoint(
            config.model.load_from,
            model,
            load_ema=config.model.resume_from.get("load_ema", False),
            null_embed_path=null_embed_path,
        )
        logger.warning(f"Missing keys: {missing}")
        logger.warning(f"Unexpected keys: {unexpected}")


    # prepare for FSDP clip grad norm calculation
    if accelerator.distributed_type == DistributedType.FSDP:
        for m in accelerator._models:
            m.clip_grad_norm_ = types.MethodType(clip_grad_norm_, m)

    # 3. build dataloader
    from omegaconf import OmegaConf
    rank = int(os.environ.get("RANK", 0))
    num_replicas = int(os.environ.get("WORLD_SIZE", 1))
    structured_config = OmegaConf.structured(config)

    # Load the dataset configuration using OmegaConf
    dataset_config = OmegaConf.load("configs/depth_dataset.yaml")

    # Convert structured_config to an unstructured DictConfig
    unstructured_config = OmegaConf.to_container(structured_config, resolve=True)
    unstructured_config = OmegaConf.create(unstructured_config)

    # Merge the dataset config into the unstructured config
    merged_config = OmegaConf.merge(unstructured_config, dataset_config)

    train_dataloader, val_dataloader = create_datasets(merged_config, rank=rank, world_size=2)
    
    train_dataloader_len = len(train_dataloader)
    logger.info(f"Number of training samples: {len(train_dataloader.dataset)}")
    logger.info(f"Total number of batches: {train_dataloader_len}")

    load_vae_feat = getattr(train_dataloader.dataset, "load_vae_feat", False)
    load_text_feat = getattr(train_dataloader.dataset, "load_text_feat", False)

    # 4. build optimizer and lr scheduler
    lr_scale_ratio = 1
    if getattr(config.train, "auto_lr", None):
        lr_scale_ratio = auto_scale_lr(
            config.train.train_batch_size * get_world_size() * config.train.gradient_accumulation_steps,
            config.train.optimizer,
            **config.train.auto_lr,
        )
    optimizer = build_optimizer(model, config.train.optimizer)
    if config.train.lr_schedule_args and config.train.lr_schedule_args.get("num_warmup_steps", None):
        config.train.lr_schedule_args["num_warmup_steps"] = (
            config.train.lr_schedule_args["num_warmup_steps"] * num_replicas
        )
    lr_scheduler = build_lr_scheduler(config.train, optimizer, train_dataloader, lr_scale_ratio)
    logger.warning(
        f"{colored(f'Basic Setting: ', 'green', attrs=['bold'])}"
        f"lr: {config.train.optimizer['lr']:.5f}, bs: {config.train.train_batch_size}, gc: {config.train.grad_checkpointing}, "
        f"gc_accum_step: {config.train.gradient_accumulation_steps}, qk norm: {config.model.qk_norm}, "
        f"fp32 attn: {config.model.fp32_attention}, attn type: {config.model.attn_type}, ffn type: {config.model.ffn_type}, "
        f"text encoder: {config.text_encoder.text_encoder_name}, captions: {config.data.caption_proportion}, precision: {config.model.mixed_precision}"
    )

    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    if accelerator.is_main_process:
        tracker_config = dict(vars(config))
        try:
            accelerator.init_trackers(args.tracker_project_name, tracker_config)
        except:
            accelerator.init_trackers(f"tb_{timestamp}")

    start_epoch = 0
    start_step = 0
    total_steps = train_dataloader_len * config.train.num_epochs

    # Resume training
    if config.model.resume_from is not None and config.model.resume_from["checkpoint"] is not None:
        rng_state = None
        ckpt_path = osp.join(config.work_dir, "checkpoints")
        check_flag = osp.exists(ckpt_path) and len(os.listdir(ckpt_path)) != 0
        if config.model.resume_from["checkpoint"] == "latest":
            if check_flag:
                checkpoints = os.listdir(ckpt_path)
                if "latest.pth" in checkpoints and osp.exists(osp.join(ckpt_path, "latest.pth")):
                    config.model.resume_from["checkpoint"] = osp.realpath(osp.join(ckpt_path, "latest.pth"))
                else:
                    checkpoints = [i for i in checkpoints if i.startswith("epoch_")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.replace(".pth", "").split("_")[3]))
                    config.model.resume_from["checkpoint"] = osp.join(ckpt_path, checkpoints[-1])
            else:
                config.model.resume_from["checkpoint"] = config.model.load_from

        if config.model.resume_from["checkpoint"] is not None:
            _, missing, unexpected, rng_state = load_checkpoint(
                **config.model.resume_from,
                model=model,
                optimizer=optimizer if check_flag else None,
                lr_scheduler=lr_scheduler if check_flag else None,
                null_embed_path=null_embed_path,
            )

            logger.warning(f"Missing keys: {missing}")
            logger.warning(f"Unexpected keys: {unexpected}")

            path = osp.basename(config.model.resume_from["checkpoint"])
        try:
            start_epoch = int(path.replace(".pth", "").split("_")[1]) - 1
            start_step = int(path.replace(".pth", "").split("_")[3])
        except:
            pass
        

        # resume randomise
        if rng_state:
            logger.info("resuming randomise")
            torch.set_rng_state(rng_state["torch"])
            torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
            np.random.set_state(rng_state["numpy"])
            random.setstate(rng_state["python"])
            generator.set_state(rng_state["generator"])  # resume generator status

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model = accelerator.prepare(model)
    optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)

    cfg.train.train_batch_size = 1
    # Start Training
    train(
        config=config,
        args=args,
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        train_diffusion=train_diffusion,
        logger=logger,
    )


if __name__ == "__main__":

    main()
