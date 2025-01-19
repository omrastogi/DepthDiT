import argparse
import logging  # Added for logging
import os

import matplotlib
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms.functional import pil_to_tensor, resize
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import time

from src.dataset import get_dataset
from src.dataset.base_depth_dataset import BaseDepthDataset, get_pred_name, DatasetMode  # noqa: F401
from src.dataset.depth_transform import get_depth_normalizer
from src.dataset.hypersim_dataset import HypersimDataset
from src.diffusion import create_diffusion
from src.util.ensemble import ensemble_depth
from src.models.models import DiT_models
from src.util.image_util import chw2hwc, colorize_depth_maps
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

'''
class DepthInference:
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.image_size = args.image_size
        self.cfg_scale = args.cfg_scale
        self.args = args
        self.ensemble_size = args.ensemble_size
        self.num_sampling_steps = args.num_sampling_steps
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.diffusion, self.vae = self.initialize_model()

    def _replace_patchembed_proj(self, model):
        """Replace the first layer to accept 8 in_channels."""
        _weight = model.x_embedder.proj.weight.clone()  # [320, 4, 3, 3]
        _bias = model.x_embedder.proj.bias.clone()  # [320]
        _weight = _weight.repeat((1, 2, 1, 1))  # Keep selected channel(s)
        _weight *= 0.5  # Half the activation magnitude
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
        logger.info("PatchEmbed projection layer has been replaced.")
        return model

    def decode_depth(self, depth_latent, vae):
        """Decode depth latent into depth map."""
        # Scale latent
        depth_latent = depth_latent / 0.18215
        z = vae.post_quant_conv(depth_latent)
        stacked = vae.decoder(z)
        depth_mean = stacked.mean(dim=1, keepdim=True)
        return depth_mean

    def get_rgb_norm(self, image_or_path):
        """Normalize the RGB image."""
        if isinstance(image_or_path, str):
            image = Image.open(image_or_path)
        elif isinstance(image_or_path, Image.Image):
            image = image_or_path
        else:
            raise ValueError("Invalid input: image_or_path must be a string or PIL Image.")
        # Resize the image to the model's expected size
        image = image.resize((self.image_size, self.image_size))
        rgb = np.asarray(image.convert("RGB"))
        rgb = np.transpose(rgb, (2, 0, 1)).astype(np.float32)  # [C, H, W]
        rgb_norm = rgb / 255.0 * 2.0 - 1.0  # [0, 255] -> [-1, 1]
        rgb_norm = torch.from_numpy(rgb_norm).float().unsqueeze(0)  # [1, C, H, W]
        return rgb_norm

    def single_infer(self, rgb_batch):
        """Run inference to generate depth maps for a batch of images."""
        device = self.device

        # Map input images to latent space and normalize latents
        with torch.no_grad():
            rgb_input_latent = self.vae.encode(rgb_batch).latent_dist.sample().mul_(0.18215)

        # Create sampling noise and classifier-free guidance
        batch_size = rgb_input_latent.shape[0]
        noise = torch.randn_like(rgb_input_latent, device=device)
        y = torch.zeros(batch_size, dtype=torch.long).to(device)

        # Prepare inputs for classifier-free guidance
        noise = torch.cat([noise, noise], 0)
        y_null = torch.tensor([1000] * batch_size, device=device)
        y = torch.cat([y, y_null], 0)
        rgb_input_latent = torch.cat([rgb_input_latent, rgb_input_latent], 0)
        model_kwargs = dict(y=y, cfg_scale=self.cfg_scale, input_img=rgb_input_latent)

        # Sample images from the diffusion model
        if self.args.scheduler == "ddim":
            samples = self.diffusion.ddim_sample_loop(
                self.model.forward_with_cfg, noise.shape, noise, clip_denoised=False,
                model_kwargs=model_kwargs, progress=False, device=device
            )
        elif self.args.scheduler == "ddpm":
            samples = self.diffusion.p_sample_loop(
                self.model.forward_with_cfg, noise.shape, noise, clip_denoised=False,
                model_kwargs=model_kwargs, progress=False, device=device
            )

        samples, _ = samples.chunk(2, dim=0)  # Discard the unconditional samples

        # Decode the depth from latent space
        with torch.no_grad():
            depth = self.decode_depth(samples, self.vae)

        # Clipping the values
        depth = torch.clamp(depth, -1.0, 1.0)

        # Normalize and colorize the depth map
        depth_pred = (depth + 1.0) / 2.0
        return depth_pred

    def pipe(self, input_image: Image.Image):
        """Process a single image through the inference pipeline."""

        # Image preprocessing
        rgb_norm = self.get_rgb_norm(input_image).to(self.device)
        input_size = input_image.size  # Original size

        # Ensemble processing
        duplicated_rgb = rgb_norm.expand(self.ensemble_size, -1, -1, -1)
        single_rgb_dataset = TensorDataset(duplicated_rgb)
        single_rgb_loader = DataLoader(
            single_rgb_dataset, batch_size=self.batch_size, shuffle=False
        )
        
        # Inference
        depth_pred_ls = []
        start_time = time.time()
        for batch in single_rgb_loader:
            (batched_img,) = batch
            if self.args.fp16:
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        depth_pred_raw = self.single_infer(
                            rgb_batch=batched_img,
                        )
            else:
                depth_pred_raw = self.single_infer(
                        rgb_batch=batched_img,
                    )
            depth_pred_ls.append(depth_pred_raw.detach())
        depth_preds = torch.concat(depth_pred_ls, dim=0)
        torch.cuda.empty_cache()  # Clear VRAM cache for ensembling
        end_time = time.time()
        print(f"Inference time: {end_time - start_time:.2f} seconds")

        # Ensemble post-processing 
        if self.ensemble_size > 1:
            depth_pred, pred_uncert = ensemble_depth(
                depth_preds,
                max_res=50,
            )
        else:
            depth_pred = depth_preds
            pred_uncert = None

        depth_pred = resize(
            depth_pred,
            input_size[::-1],
            antialias=True,
        )

        # Image Postprocessing
        depth_pred = depth_pred.squeeze()
        depth_pred = depth_pred.cpu().numpy()

        if pred_uncert is not None:
            pred_uncert = pred_uncert.squeeze().cpu().numpy()

        depth_pred = depth_pred.clip(0, 1)   # Clip output range
        
        depth_colored = colorize_depth_maps(  # Colorize
            depth_pred, 0, 1, cmap="Spectral"
        ).squeeze()  # [3, H, W], values in (0, 1)

        depth_colored = (depth_colored * 255).astype(np.uint8)
        depth_colored_hwc = chw2hwc(depth_colored)
        depth_colored_img = Image.fromarray(depth_colored_hwc)
        
        return depth_pred, depth_colored_img, pred_uncert

    def initialize_model(self):
        """Initialize the model, diffusion, and VAE."""
        device = self.device

        # Load model
        latent_size = self.args.image_size // 8
        model = DiT_models[self.args.model](
            input_size=latent_size,
            num_classes=self.args.num_classes,
        ).to(device)

        # Modify the model architecture before loading state dict
        if 8 != model.x_embedder.proj.weight.shape[1]:
            model = self._replace_patchembed_proj(model)

        # Load checkpoint
        ckpt_path = self.args.ckpt
        checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        if "model" in checkpoint:  # Supports checkpoints from train.py
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
        model.eval()

        # Initialize diffusion and VAE
        diffusion = create_diffusion(str(self.args.num_sampling_steps))
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{self.args.vae}").to(device)


        logger.info("Model, diffusion, and VAE have been initialized.")
        if self.args.fp16:
            model = model.to(device).half()
            vae = vae.to(device).half()

        return model, diffusion, vae
'''

import torch
from torch import nn
from torch.nn import Conv2d, Parameter
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

# Note: The following dependencies need to be defined or imported:
# - DiT_models: a dictionary mapping model names to model classes
# - create_diffusion: a function to create a diffusion process
# - AutoencoderKL: from 'diffusers' library
# - ensemble_depth: function to ensemble depth predictions
# - resize: function to resize tensors or images
# - colorize_depth_maps: function to colorize depth maps
# - chw2hwc: function to convert from CHW to HWC format

class DepthPipeline:
    def __init__(
        self,
        batch_size=1,
        image_size=512,
        cfg_scale=4.0,
        ensemble_size=1,
        num_sampling_steps=50,
        num_classes=1000,
        scheduler='ddim',
        fp16=False,
        model=None,
        vae=None,
        diffusion=None,
        ckpt_path='path/to/checkpoint.ckpt',
        vae_name='2-0',
        model_name='DiT-XL-2',

    ):
        self.batch_size = batch_size
        self.image_size = image_size
        self.cfg_scale = cfg_scale
        self.ensemble_size = ensemble_size
        self.num_sampling_steps = num_sampling_steps
        self.model_name = model_name
        self.num_classes = num_classes
        self.ckpt_path = ckpt_path
        self.vae_name = vae_name
        self.scheduler = scheduler
        self.fp16 = fp16
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if model is None:
            self.model, self.diffusion, self.vae = self.initialize_model()
        else:
            self.model, self.diffusion, self.vae = model, diffusion, vae

    def _replace_patchembed_proj(self, model):
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
        logger.info("PatchEmbed projection layer has been replaced.")
        return model

    def decode_depth(self, depth_latent, vae):
        """Decode depth latent into depth map."""
        depth_latent = depth_latent / 0.18215
        z = vae.post_quant_conv(depth_latent)
        stacked = vae.decoder(z)
        depth_mean = stacked.mean(dim=1, keepdim=True)
        return depth_mean

    def get_rgb_norm(self, image_or_path):
        """Normalize the RGB image."""
        if isinstance(image_or_path, str):
            image = Image.open(image_or_path)
        elif isinstance(image_or_path, Image.Image):
            image = image_or_path
        else:
            raise ValueError("Invalid input: image_or_path must be a string or PIL Image.")
        image = image.resize((self.image_size, self.image_size))
        rgb = np.asarray(image.convert("RGB"))
        rgb = np.transpose(rgb, (2, 0, 1)).astype(np.float32)
        rgb_norm = rgb / 255.0 * 2.0 - 1.0
        rgb_norm = torch.from_numpy(rgb_norm).float().unsqueeze(0)
        return rgb_norm

    def single_infer(self, rgb_batch, task_emb=None):
        """Run inference to generate depth maps for a batch of images."""
        device = self.device

        with torch.no_grad():
            rgb_input_latent = self.vae.encode(rgb_batch).latent_dist.sample().mul_(0.18215)

        batch_size = rgb_input_latent.shape[0]
        noise = torch.randn_like(rgb_input_latent, device=device)
        y = torch.zeros(batch_size, dtype=torch.long).to(device)

        noise = torch.cat([noise, noise], 0)
        y_null = torch.tensor([1000] * batch_size, device=device)
        y = torch.cat([y, y_null], 0)
        rgb_input_latent = torch.cat([rgb_input_latent, rgb_input_latent], 0)
        model_kwargs = dict(y=y, cfg_scale=self.cfg_scale, input_img=rgb_input_latent, task_emb=task_emb)

        if self.scheduler == "ddim":
            samples = self.diffusion.ddim_sample_loop(
                self.model.forward_with_cfg, noise.shape, noise, clip_denoised=False,
                model_kwargs=model_kwargs, progress=False, device=device
            )
        elif self.scheduler == "ddpm":
            samples = self.diffusion.p_sample_loop(
                self.model.forward_with_cfg, noise.shape, noise, clip_denoised=False,
                model_kwargs=model_kwargs, progress=False, device=device
            )

        samples, _ = samples.chunk(2, dim=0)

        with torch.no_grad():
            depth = self.decode_depth(samples, self.vae)

        depth = torch.clamp(depth, -1.0, 1.0)
        depth_pred = (depth + 1.0) / 2.0
        return depth_pred

    def pipe(self, input_image: Image.Image):
        """Process a single image through the inference pipeline."""
        rgb_norm = self.get_rgb_norm(input_image).to(self.device)
        input_size = input_image.size

        duplicated_rgb = rgb_norm.expand(self.ensemble_size, -1, -1, -1)
        single_rgb_dataset = TensorDataset(duplicated_rgb)
        single_rgb_loader = DataLoader(
            single_rgb_dataset, batch_size=self.batch_size, shuffle=False
        )
        
        import time

        depth_pred_ls = []
        start_time = time.time()
        for batch in single_rgb_loader:
            (batched_img,) = batch
            if self.fp16:
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        depth_pred_raw = self.single_infer(rgb_batch=batched_img)
            else:
                depth_pred_raw = self.single_infer(rgb_batch=batched_img)
            depth_pred_ls.append(depth_pred_raw.detach())
        depth_preds = torch.concat(depth_pred_ls, dim=0)
        torch.cuda.empty_cache()
        end_time = time.time()
        print(f"Inference time: {end_time - start_time:.2f} seconds")

        if self.ensemble_size > 1:
            depth_pred, pred_uncert = ensemble_depth(depth_preds, max_res=50)
        else:
            depth_pred = depth_preds
            pred_uncert = None

        depth_pred = resize(depth_pred, input_size[::-1], antialias=True)

        depth_pred = depth_pred.squeeze().cpu().numpy()

        if pred_uncert is not None:
            pred_uncert = pred_uncert.squeeze().cpu().numpy()

        depth_pred = depth_pred.clip(0, 1)
        
        depth_colored = colorize_depth_maps(depth_pred, 0, 1, cmap="Spectral").squeeze()
        depth_colored = (depth_colored * 255).astype(np.uint8)
        depth_colored_hwc = chw2hwc(depth_colored)
        depth_colored_img = Image.fromarray(depth_colored_hwc)
        
        return depth_pred, depth_colored_img, pred_uncert
    
    def pipe_task_cond(self, input_image: Image.Image, task_emb):
        """Process a single image through the inference pipeline."""
        rgb_norm = self.get_rgb_norm(input_image).to(self.device)
        input_size = input_image.size

        duplicated_rgb = rgb_norm.expand(self.ensemble_size, -1, -1, -1)
        single_rgb_dataset = TensorDataset(duplicated_rgb)
        single_rgb_loader = DataLoader(
            single_rgb_dataset, batch_size=self.batch_size, shuffle=False
        )
        
        import time

        depth_pred_ls = []
        start_time = time.time()
        for batch in single_rgb_loader:
            (batched_img,) = batch
            if self.fp16:
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        depth_pred_raw = self.single_infer(rgb_batch=batched_img, task_emb=task_emb)
            else:
                depth_pred_raw = self.single_infer(rgb_batch=batched_img, task_emb=task_emb)
            depth_pred_ls.append(depth_pred_raw.detach())
        depth_preds = torch.concat(depth_pred_ls, dim=0)
        torch.cuda.empty_cache()
        end_time = time.time()
        print(f"Inference time: {end_time - start_time:.2f} seconds")

        if self.ensemble_size > 1:
            depth_pred, pred_uncert = ensemble_depth(depth_preds, max_res=50)
        else:
            depth_pred = depth_preds
            pred_uncert = None

        depth_pred = resize(depth_pred, input_size[::-1], antialias=True)

        depth_pred = depth_pred.squeeze().cpu().numpy()

        if pred_uncert is not None:
            pred_uncert = pred_uncert.squeeze().cpu().numpy()

        depth_pred = depth_pred.clip(0, 1)
        
        depth_colored = colorize_depth_maps(depth_pred, 0, 1, cmap="Spectral").squeeze()
        depth_colored = (depth_colored * 255).astype(np.uint8)
        depth_colored_hwc = chw2hwc(depth_colored)
        depth_colored_img = Image.fromarray(depth_colored_hwc)
        
        return depth_pred, depth_colored_img, pred_uncert


    def initialize_model(self):
        """Initialize the model, diffusion, and VAE."""
        device = self.device

        latent_size = self.image_size // 8
        model = DiT_models[self.model_name](
            input_size=latent_size,
            num_classes=self.num_classes,
        ).to(device)

        if 8 != model.x_embedder.proj.weight.shape[1]:
            model = self._replace_patchembed_proj(model)

        ckpt_path = self.ckpt_path
        checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
        model.eval()

        diffusion = create_diffusion(str(self.num_sampling_steps))
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{self.vae_name}").to(device)

        logger.info("Model, diffusion, and VAE have been initialized.")
        if self.fp16:
            model = model.to(device).half()
            vae = vae.to(device).half()

        return model, diffusion, vae

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=512)
    parser.add_argument("--num-classes", type=int, default=1000) #TODO: Not required
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for processing images.")
    parser.add_argument("--save-npy", action="store_true", help="Save depth as npy")
    parser.add_argument("--ensemble-size", type=int, default=10, help="Size of the ensemble for depth prediction.")
    parser.add_argument("--image-path", type=str, required=True, help="Path to the input image or directory.")
    parser.add_argument("--scheduler", type=str, choices=["ddim", "ddpm"], default="ddim", help="Scheduler type to use for inference.")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision for inference.")
    args = parser.parse_args()
    depth_inference = DepthInference(
        batch_size=args.batch_size,
        image_size=args.image_size,
        cfg_scale=args.cfg_scale,
        ensemble_size=args.ensemble_size,
        num_sampling_steps=args.num_sampling_steps,
        model_name=args.model,
        num_classes=args.num_classes,
        ckpt_path=args.ckpt,
        vae_name=args.vae,
        scheduler=args.scheduler,
        fp16=args.fp16,
    )
    input_image = Image.open(args.image_path).convert('RGB')
    depth_pred, depth_colored_img, pred_uncert = depth_inference.pipe(input_image)





"""
python pipeline.py \
--batch-size 10 \
--num-sampling-steps 50 \
--ensemble-size 10 \
--fp16 \
--image-path /mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/depth_estimation_experiments/DiT/data/hypersim_vis/rgb_cam_00_fr0002.png \
--ckpt /mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/depth_estimation_experiments/DiT/checkpoints/model_vkitti_hypersim_4_epoch_multires/checkpoints/0014000.pt \
--output-dir /mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/depth_estimation_experiments/DiT

"""