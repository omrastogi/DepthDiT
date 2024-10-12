import torch
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from torchvision.transforms.functional import resize
from torch.utils.data import DataLoader
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse

from dataset.base_depth_dataset import BaseDepthDataset, get_pred_name, DatasetMode  # noqa: F401
from dataset import get_dataset
from dataset.hypersim_dataset import HypersimDataset
from dataset.depth_transform import get_depth_normalizer
from types import SimpleNamespace
import os
import numpy as np
import matplotlib
from PIL import Image
from omegaconf import OmegaConf

from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms.functional import pil_to_tensor
from ensemble import ensemble_depth
import logging  # Added for logging
from tqdm import tqdm


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def chw2hwc(chw):
    """Convert a tensor from CHW format to HWC format."""
    assert 3 == len(chw.shape)
    if isinstance(chw, torch.Tensor):
        hwc = torch.permute(chw, (1, 2, 0))
    elif isinstance(chw, np.ndarray):
        hwc = np.moveaxis(chw, 0, -1)
    return hwc

def colorize_depth_maps(depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None):
    """Colorize depth maps."""
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().squeeze().cpu().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()

    # Reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # Colorize depth map
    cm = matplotlib.cm.get_cmap(cmap)
    depth_normalized = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    depth_colored_np = cm(depth_normalized)[:, :, :, 0:3]  # Values from 0 to 1
    depth_colored_np = np.moveaxis(depth_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().cpu().numpy()
        valid_mask = valid_mask.squeeze()
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        depth_colored_np[~valid_mask] = 0

    if isinstance(depth_map, torch.Tensor):
        depth_colored = torch.from_numpy(depth_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        depth_colored = depth_colored_np
    return depth_colored

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

    def load_images(self, path):
        """Load a single image or all images from a directory."""
        image_tensors = []
        if os.path.isdir(path):
            # Load all images from the directory
            for filename in os.listdir(path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(path, filename)
                    logger.info(f"Loading image: {image_path}")
                    image = Image.open(image_path)
                    original_shape = image.size  # (width, height)
                    image_tensor = self.get_rgb_norm(image).to(self.device)
                    image_tensors.append((image_tensor, original_shape, filename))
        else:
            # Load the single image
            image = Image.open(path)
            original_shape = image.size  # (width, height)
            image_tensor = self.get_rgb_norm(image).to(self.device)
            filename = os.path.basename(path)
            image_tensors.append((image_tensor, original_shape, filename))

        return image_tensors

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
        rgb_norm = self.get_rgb_norm(input_image).to(self.device)
        input_size = input_image.size  # Original size

        # Ensemble processing
        duplicated_rgb = rgb_norm.expand(self.ensemble_size, -1, -1, -1)
        single_rgb_dataset = TensorDataset(duplicated_rgb)

        single_rgb_loader = DataLoader(
            single_rgb_dataset, batch_size=self.batch_size, shuffle=False
        )
        depth_pred_ls = []
        for batch in single_rgb_loader:
            (batched_img,) = batch
            depth_pred_raw = self.single_infer(
                rgb_batch=batched_img,
            )
            depth_pred_ls.append(depth_pred_raw.detach())
        depth_preds = torch.concat(depth_pred_ls, dim=0)
        torch.cuda.empty_cache()  # Clear VRAM cache for ensembling

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

        depth_pred = depth_pred.squeeze()
        depth_pred = depth_pred.cpu().numpy()

        if pred_uncert is not None:
            pred_uncert = pred_uncert.squeeze().cpu().numpy()

        # Clip output range
        depth_pred = depth_pred.clip(0, 1)

        # Colorize
        depth_colored = colorize_depth_maps(
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
        return model, diffusion, vae

def process_image(args):
    """Process images and save depth predictions."""
    depth_inference = DepthInference(args)
    cfg_data = OmegaConf.load(args.dataset_config)

    dataset: BaseDepthDataset = get_dataset(
        cfg_data, base_data_dir=args.base_data_dir, mode=DatasetMode.RGB_ONLY
    )

    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    with torch.no_grad():
        for batch in tqdm(
            dataloader, desc=f"Inferencing on {dataset.disp_name}", leave=True
        ):
            rgb_int = batch["rgb_int"].squeeze().numpy().astype(np.uint8)  # [3, H, W]
            rgb_int = np.moveaxis(rgb_int, 0, -1)  # [H, W, 3]
            input_image = Image.fromarray(rgb_int)
            depth_pred, colored_depth, pred_uncert = depth_inference.pipe(input_image)

            # Save predictions
            rgb_filename = batch["rgb_relative_path"][0]
            rgb_basename = os.path.basename(rgb_filename)
            scene_dir = os.path.join(args.output_dir, os.path.dirname(rgb_filename))
            if not os.path.exists(scene_dir):
                os.makedirs(scene_dir, exist_ok=True)
            pred_basename = get_pred_name(
                rgb_basename, dataset.name_mode, suffix=".npy"
            )
            save_to = os.path.join(scene_dir, pred_basename)
            if os.path.exists(save_to):
                logger.warning(f"Existing file: '{save_to}' will be overwritten.")
            np.save(save_to, depth_pred)

            # Save colored depth as PNG
            colored_depth_basename = get_pred_name(
                rgb_basename, dataset.name_mode, suffix=".png"
            )
            colored_depth_save_to = os.path.join(scene_dir, colored_depth_basename)
            colored_depth.save(colored_depth_save_to)
            logger.info(f"Saved colored depth to {colored_depth_save_to}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000) #TODO: Not required
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for processing images.")
    parser.add_argument("--save-npy", action="store_true", help="Save depth as npy")
    parser.add_argument("--ensemble-size", type=int, default=10, help="Size of the ensemble for depth prediction.")
    parser.add_argument("--dataset-config", type=str, required=True, help="Path to config file of evaluation dataset.")
    parser.add_argument("--base-data-dir", type=str, required=True, help="Path to base data directory.")

    args = parser.parse_args()
    process_image(args)


"""
python infer.py \
  --model DiT-XL/2 \
  --image-size 512 \
  --batch-size 10 \
  --num-sampling-steps 5\
  --ensemble-size 1 \
  --dataset-config config/new_benchmark_dataset/data_nyu_test.yaml \
  --base-data-dir /mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/Marigold/eval_dataset \
  --ckpt /mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/DiT/results/6-epochs/checkpoints/0014000.pt \
  --output-dir /mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/DiT/results/batch_eval/nyu_test/prediction 
"""
