# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from src.models.models import DiT_models
import argparse

from dataset.base_depth_dataset import BaseDepthDataset, get_pred_name, DatasetMode  # noqa: F401
from dataset.hypersim_dataset import HypersimDataset
from dataset.depth_transform import get_depth_normalizer
from types import SimpleNamespace
import os
import numpy as np
import matplotlib
from PIL import Image



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

def validation(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    ).to(device)

    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    
    if args.depth_ckpt == True:
        model = _replace_patchembed_proj(model)
    model.load_state_dict(state_dict, strict=False)
    model.eval()  # important!
    
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    
    if 8 != model.x_embedder.proj.weight.shape[1]:
        model = _replace_patchembed_proj(model) 
    
    # Setup dataset params
    cfg_normalizer = SimpleNamespace(
        type='scale_shift_depth', clip=True, norm_min=-1.0, norm_max=1.0, min_max_quantile=0.02
    )

    depth_transform = get_depth_normalizer(cfg_normalizer=cfg_normalizer)
    kwargs = {'augmentation_args': {'lr_flip_p': 0.5}, 'depth_transform': depth_transform}
    kwargs['augmentation_args'] = SimpleNamespace(**kwargs['augmentation_args'])

    cfg_data_split = {
        'name': 'hypersim',
        'disp_name': 'hypersim_train',
        'dir': 'Hypersim/processed/train',
        'filenames': 'data_split/hypersim/filename_list_train_filtered_subset.txt',
        'resize_to_hw': [512, 512] # Marigold code resizes it to [480, 640]
    }

    # Calling dataset 
    dataset = HypersimDataset(
        mode=DatasetMode.TRAIN,
        filename_ls_path="/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/DiT/data_split/hypersim/selected_vis_sample.txt",
        dataset_dir=os.path.join("/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/Marigold/data", "Hypersim/processed/val"),
        **cfg_data_split,
        **kwargs,
    )
    
    batch_size = 1
    cfg_scale = 4.0
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    # Get the next batch
    batch = next(iter(loader))
    print(batch.keys())
    rgb = batch["rgb_norm"].to(device)
    rgb_int = batch["rgb_int"].to(device)  # Real RGB images from batch

    # Zero the class-conditioning
    y = torch.zeros(batch_size, dtype=torch.long).to(device)

    print(rgb.shape)
    with torch.no_grad(): # Map input images to latent space + normalize latents:
        rgb_input_latent = vae.encode(rgb).latent_dist.sample().mul_(0.18215)

    noise = torch.randn_like(rgb_input_latent, device=device)

    # Setup classifier-free guidance:
    noise = torch.cat([noise, noise], 0)
    y_null = torch.tensor([1000] * batch_size, device=device)
    y = torch.cat([y, y_null], 0)
    rgb_input_latent = torch.cat([rgb_input_latent, rgb_input_latent], 0) # concatenating this as well
    model_kwargs = dict(y=y, cfg_scale=cfg_scale, input_img=rgb_input_latent)

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )

    # Remove null class samples
    samples, _ = samples.chunk(2, dim=0)  
    
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
    print(depth_colored.shape)
    depth_colored = np.transpose(depth_colored, (1, 2, 0))

    Image.fromarray(depth_colored).save("depth_image.png")


def chw2hwc(chw):
    assert 3 == len(chw.shape)
    if isinstance(chw, torch.Tensor):
        hwc = torch.permute(chw, (1, 2, 0))
    elif isinstance(chw, np.ndarray):
        hwc = np.moveaxis(chw, 0, -1)
    return hwc


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

def decode_depth(depth_latent, vae):
    """
    Decode depth latent into depth map.

    Args:
        depth_latent (`torch.Tensor`):
            Depth latent to be decoded.

    Returns:
        `torch.Tensor`: Decoded depth map.
    """
    # scale latent
    depth_latent = depth_latent / 0.18215
    # decode
    z = vae.post_quant_conv(depth_latent)
    stacked = vae.decoder(z)
    # mean of output channels
    depth_mean = stacked.mean(dim=1, keepdim=True)
    return depth_mean

def get_rgb_norm(image_path):
    # Load the image using PIL
    image = Image.open(image_path)
    # Resize the image to 512x512
    image = image.resize((512, 512))
    # Convert the image to a numpy array and ensure it's in RGB format
    rgb = np.asarray(image.convert("RGB"))
    # Transpose the image to get it in [C, H, W] format, as expected by the code
    rgb = np.transpose(rgb, (2, 0, 1)).astype(int)  # [rgb, H, W]
    # Normalize the RGB values to the range [-1, 1]
    rgb_norm = rgb / 255.0 * 2.0 - 1.0  # [0, 255] -> [-1, 1]
    # Convert the numpy array to a torch tensor and add a batch dimension
    rgb_norm = torch.from_numpy(rgb_norm).float().unsqueeze(0)  # [1, C, H, W]
    return rgb_norm

def inference(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    ).to(device)
    
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    if args.depth_ckpt == True:
        model = _replace_patchembed_proj(model)
    model.load_state_dict(state_dict, strict=False)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    rgb = get_rgb_norm("/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/Marigold/data/Hypersim/processed/val/ai_015_004/rgb_cam_00_fr0002.png").to(device)
    with torch.no_grad(): # Map input images to latent space + normalize latents:
        rgb_input_latent = vae.encode(rgb).latent_dist.sample().mul_(0.18215)

    n = rgb_input_latent.shape[0]

    if 8 != model.x_embedder.proj.weight.shape[1]:
        model = _replace_patchembed_proj(model) 
        # model.in_channels = 8

    # Create sampling noise:
    n = rgb_input_latent.shape[0]
    noise = torch.randn_like(rgb_input_latent, device=device)
    y = torch.zeros(n, dtype=torch.long).to(device)

    # Setup classifier-free guidance:
    noise = torch.cat([noise, noise], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    rgb_input_latent = torch.cat([rgb_input_latent, rgb_input_latent], 0) # concating this as well
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale, input_img=rgb_input_latent)


    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
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
    depth_colored = np.transpose(depth_colored, (1, 2, 0))


    # Save and display images as numpy array:
    depth_colored_image = Image.fromarray(depth_colored)
    depth_colored_image.save("sample.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--depth-ckpt", type=bool, default=True, help="Boolean flag to indicate if depth checkpoint should be used.")
    args = parser.parse_args()
    # validation(args)
    inference(args)

"""
python depth_dit.py --model DiT-XL/2 --image-size 512 --ckpt /mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/DiT/results/1-epoch-not-valid-mask/checkpoints/0002400.pt
"""