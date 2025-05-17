# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import os
import re
import subprocess
import tarfile
import time
import warnings
from omegaconf import OmegaConf
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

# from datetime import datetime
from typing import List, Optional

import pyrallis
import torch
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from termcolor import colored
from torchvision.utils import save_image
from tqdm import tqdm

warnings.filterwarnings("ignore")  # ignore warning

from diffusion import DPMS, FlowEuler, SASolverSampler
from diffusion.data.datasets.utils import ASPECT_RATIO_512_TEST, ASPECT_RATIO_1024_TEST, ASPECT_RATIO_2048_TEST
from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder, get_vae, vae_decode, vae_encode
from diffusion.model.utils import prepare_prompt_ar
from diffusion.utils.config import SanaConfig
from diffusion.utils.logger import get_root_logger
from tools.download import find_model

from src.dataset.base_depth_dataset import BaseDepthDataset, get_pred_name, DatasetMode
from src.dataset import get_dataset
from src.utils.image_utils import colorize_depth_maps, chw2hwc

from diffusion.data.datasets.utils import ASPECT_RATIO_256, ASPECT_RATIO_512, ASPECT_RATIO_1024, ASPECT_RATIO_2048, ASPECT_RATIO_2880, ASPECT_RATIO_4096


def set_env(seed=0, latent_size=256):
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    for _ in range(30):
        torch.randn(1, 4, latent_size, latent_size)


def get_dict_chunks(data, bs):
    keys = []
    for k in data:
        keys.append(k)
        if len(keys) == bs:
            yield keys
            keys = []
    if keys:
        yield keys


def create_tar(data_path):
    tar_path = f"{data_path}.tar"
    with tarfile.open(tar_path, "w") as tar:
        tar.add(data_path, arcname=os.path.basename(data_path))
    print(f"Created tar file: {tar_path}")
    return tar_path


def delete_directory(exp_name):
    if os.path.exists(exp_name):
        subprocess.run(["rm", "-r", exp_name], check=True)
        print(f"Deleted directory: {exp_name}")


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
def inference(config, model, val_dataloader, device, vae=None):
    torch.cuda.empty_cache()
    model.eval()
    vis_sampler = config.scheduler.vis_sampler
    print("Inference started")
    logger.info(colored(f"Saving images at {config.output_dir}", "green"))
    for idx, batch in enumerate(val_dataloader):
        rgb = batch["rgb_norm"].to(device).to(torch.float16)
        rgb_int = batch["rgb_int"].to(device).to(torch.float16)  # Real RGB images from batch

        # Map input image to latent space + normalize latents
        input_latent = vae_encode(config.vae.vae_type, vae, rgb, config.vae.sample_posterior, device)
        latent_size_h, latent_size_w = input_latent.shape[2], input_latent.shape[3]
        z = torch.randn(1, config.vae.vae_latent_dim, latent_size_h, latent_size_w, device=device)

        # Embedding preparation
        emb_masks = null_caption_token.attention_mask
        caption_embs = null_caption_embs
        null_y = null_caption_embs.repeat(1, 1, 1)

        input_latent = torch.cat([input_latent] * 2)

        model_kwargs = {
            'data_info': None,
            'mask': emb_masks,
            'input_latent': input_latent
        }

        # Sampling based on the specified vis_sampler
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

        # Decode the depth from latent space
        depth = decode_depth(config.vae.vae_type, vae, latent)
        depth = torch.clip(depth, -1.0, 1.0)

        # Convert depth to colored depth map
        depth_colored = convert_depth_to_colored(depth)

        # Save predictions
        rgb_filename = batch["rgb_relative_path"][0]
        rgb_basename = os.path.basename(rgb_filename)
        scene_dir = os.path.join(config.output_dir, os.path.dirname(rgb_filename))
        if not os.path.exists(scene_dir):
            os.makedirs(scene_dir, exist_ok=True)

        # Save depth prediction as .npy
        pred_basename = get_pred_name(rgb_basename, dataset.name_mode, suffix=".npy")
        save_to = os.path.join(scene_dir, pred_basename)
        np.save(save_to, depth.squeeze().detach().cpu().numpy())
        logger.info(f"Depth prediction saved to: {save_to}")

        # Save colored depth as .png
        colored_depth_basename = get_pred_name(rgb_basename, dataset.name_mode, suffix=".png")
        colored_depth_save_to = os.path.join(scene_dir, colored_depth_basename)
        Image.fromarray(depth_colored).save(colored_depth_save_to)
        logger.info(f"Colored depth map saved to: {colored_depth_save_to}")

        print(f"Saved depth and colored depth maps for image {idx + 1}/{len(val_dataloader)}")

        # Clean up
        del z, input_latent, latent, depth
        del depth_colored
        del null_y, caption_embs, emb_masks

    print("Inference completed.")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config")
    return parser.parse_known_args()[0]


@dataclass
class SanaDepthInference(SanaConfig):
    model_config: Optional[str] = "configs/sana_config/1024ms/Sana_1600M_img1024.yaml"  # config
    model_path: Optional[str] = "hf://Efficient-Large-Model/Sana_1600M_1024px/checkpoints/Sana_1600M_1024px.pth"
    data_config: str = "configs/dataset/data_nyu_test.yaml"
    output_dir: str = "output/inference"
    base_data_dir: str = None
    version: str = "sigma"
    sample_nums: int = 100_000
    bs: int = 1
    cfg_scale: float = 4.5
    pag_scale: float = 1.0
    sampling_algo: str = "flow_dpm-solver"
    seed: int = 0
    dataset: str = "custom"
    step: int = -1
    add_label: str = ""
    tar_and_del: bool = False
    exist_time_prefix: str = ""
    gpu_id: int = 0
    custom_image_size: Optional[int] = None
    start_index: int = 0
    end_index: int = 30_000
    interval_guidance: List[float] = field(default_factory=lambda: [0, 1])
    ablation_selections: Optional[List[float]] = None
    ablation_key: Optional[str] = None
    debug: bool = False
    if_save_dirname: bool = False


if __name__ == "__main__":
    args = get_args()
    config = args = pyrallis.parse(config_class=SanaDepthInference, config_path=args.config)

    args.image_size = config.model.image_size
    if args.custom_image_size:
        args.image_size = args.custom_image_size
        print(f"custom_image_size: {args.image_size}")

    set_env(args.seed, args.image_size // config.vae.vae_downsample_rate)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = get_root_logger()

    # only support fixed latent size currently
    latent_size = args.image_size // config.vae.vae_downsample_rate
    max_sequence_length = config.text_encoder.model_max_length
    pe_interpolation = config.model.pe_interpolation
    micro_condition = config.model.micro_condition
    flow_shift = config.scheduler.flow_shift
    pag_applied_layers = config.model.pag_applied_layers
    guidance_type = "classifier-free_PAG"
    assert (
        isinstance(args.interval_guidance, list)
        and len(args.interval_guidance) == 2
        and args.interval_guidance[0] <= args.interval_guidance[1]
    )
    args.interval_guidance = [max(0, args.interval_guidance[0]), min(1, args.interval_guidance[1])]
    sample_steps_dict = {"dpm-solver": 20, "sa-solver": 25, "flow_dpm-solver": 20, "flow_euler": 28}
    sample_steps = args.step if args.step != -1 else sample_steps_dict[args.sampling_algo]
    if config.model.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif config.model.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    elif config.model.mixed_precision == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(f"weigh precision {config.model.mixed_precision} is not defined")
    logger.info(f"Inference with {weight_dtype}, default guidance_type: {guidance_type}, flow_shift: {flow_shift}")

    vae = get_vae(config.vae.vae_type, config.vae.vae_pretrained, device).to(weight_dtype)
    tokenizer, text_encoder = get_tokenizer_and_text_encoder(name=config.text_encoder.text_encoder_name, device=device)

    null_caption_token = tokenizer(
        "", max_length=max_sequence_length, padding="max_length", truncation=True, return_tensors="pt"
    ).to(device)
    null_caption_embs = text_encoder(null_caption_token.input_ids, null_caption_token.attention_mask)[0]

    # model setting
    pred_sigma = getattr(config.scheduler, "pred_sigma", True)
    learn_sigma = getattr(config.scheduler, "learn_sigma", True) and pred_sigma
    model_kwargs = {
        "pe_interpolation": config.model.pe_interpolation,
        "config": config,
        "model_max_length": config.text_encoder.model_max_length,
        "qk_norm": config.model.qk_norm,
        "micro_condition": config.model.micro_condition,
        "caption_channels": text_encoder.config.hidden_size,
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
        "depth_channels": 32,
    }
    model = build_model(
        config.model.model, use_fp32_attention=config.model.get("fp32_attention", False), **model_kwargs
    ).to(device)
    # model = build_model(config.model, **model_kwargs).to(device)
    logger.info(
        f"{model.__class__.__name__}:{config.model.model}, Model Parameters: {sum(p.numel() for p in model.parameters()):,}"
    )
    logger.info("Generating sample from ckpt: %s" % args.model_path)
    state_dict = find_model(args.model_path)
    if "pos_embed" in state_dict["state_dict"]:
        del state_dict["state_dict"]["pos_embed"]
        
    
    depth = True
    if depth:
        if model.x_embedder.proj.weight.shape[1] != state_dict["state_dict"]['x_embedder.proj.weight'].shape[1]:
            if state_dict["state_dict"]['x_embedder.proj.weight'].shape[1] * 2 == model.x_embedder.proj.weight.shape[1]:
                state_dict["state_dict"]['x_embedder.proj.weight'] = state_dict["state_dict"]['x_embedder.proj.weight'].repeat((1, 2, 1, 1))
            else:
                raise ValueError(
                    f"Shape mismatch: Model weight shape {model.x_embedder.proj.weight.shape[1]} is not compatible with "
                    f"state_dict weight shape {state_dict['x_embedder.proj.weight'].shape[1]} for repetition."
                )
                
    missing, unexpected = model.load_state_dict(state_dict["state_dict"], strict=False)
    logger.warning(f"Missing keys: {missing}")
    logger.warning(f"Unexpected keys: {unexpected}")
    model.eval().to(weight_dtype)
    base_ratios = eval(f"ASPECT_RATIO_{args.image_size}_TEST")
    args.sampling_algo = (
        args.sampling_algo
        if ("flow" not in args.model_path or args.sampling_algo == "flow_dpm-solver")
        else "flow_euler"
    )
        
    aspect_ratio_map = globals().get(config.model.aspect_ratio_type, None)
    cfg_data = OmegaConf.load(args.data_config)
    
    dataset: BaseDepthDataset = get_dataset(
        cfg_data, 
        base_data_dir=args.base_data_dir, 
        mode=DatasetMode.RGB_ONLY,
        aspect_ratio_map=aspect_ratio_map
    )
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    match = re.search(r".*epoch_(\d+).*step_(\d+).*", args.model_path)
    epoch_name, step_name = match.groups() if match else ("unknown", "unknown")

    

    # def create_save_root(args, dataset, epoch_name, step_name, sample_steps, guidance_type):
    #     save_root = os.path.join(
    #         img_save_dir,
    #         # f"{datetime.now().date() if args.exist_time_prefix == '' else args.exist_time_prefix}_"
    #         f"{dataset}_epoch{epoch_name}_step{step_name}_scale{args.cfg_scale}"
    #         f"_step{sample_steps}_size{args.image_size}_bs{args.bs}_samp{args.sampling_algo}"
    #         f"_seed{args.seed}_{str(weight_dtype).split('.')[-1]}",
    #     )
    #     if args.pag_scale != 1.0:
    #         save_root = save_root.replace(f"scale{args.cfg_scale}", f"scale{args.cfg_scale}_pagscale{args.pag_scale}")
    #     if flow_shift != 1.0:
    #         save_root += f"_flowshift{flow_shift}"
    #     if guidance_type != "classifier-free":
    #         save_root += f"_{guidance_type}"
    #     if args.interval_guidance[0] != 0 and args.interval_guidance[1] != 1:
    #         save_root += f"_intervalguidance{args.interval_guidance[0]}{args.interval_guidance[1]}"
    #     save_root += f"_imgnums{args.sample_nums}" + args.add_label
    #     return save_root

    def guidance_type_select(default_guidance_type, pag_scale, attn_type):
        guidance_type = default_guidance_type
        if not (pag_scale > 1.0 and attn_type == "linear"):
            logger.info("Setting back to classifier-free")
            guidance_type = "classifier-free"
        return guidance_type
    
    inference(config, model, dataloader, device, vae=vae)
    print(
        colored(f"Sana inference has finished. Results stored at ", "green"),
        colored(f"{config.output_dir}", attrs=["bold"]),
        ".",
    )
