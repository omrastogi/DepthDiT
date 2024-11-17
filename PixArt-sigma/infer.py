
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
import torch.nn.functional as F

from diffusers.models import AutoencoderKL
from tqdm import tqdm

from src.dataset import get_dataset
from src.dataset.base_depth_dataset import BaseDepthDataset, get_pred_name, DatasetMode  # noqa: F401
from src.dataset.depth_transform import get_depth_normalizer
from src.dataset.hypersim_dataset import HypersimDataset

# from src.diffusion import create_diffusion
from src.utils.ensemble import ensemble_depth
from diffusion.model.nets import PixArtMS_XL_2, PixArt_XL_2
from diffusion import IDDPM, DPMS, SASolverSampler
from src.utils.embedding_utils import save_null_caption_embeddings, load_null_caption_embeddings
import diffusion.data.datasets.utils as ds_utils


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_default_hw(ar_input, ratios, device='cpu'):
    """
    Finds the closest aspect ratio from the provided ratios dictionary and returns the default height and width.

    Args:
        ar_input (str): Aspect ratio in the format 'width:height' (e.g., '16:9').
        ratios (dict): A dictionary where keys are aspect ratios (as strings) 
                       and values are [height, width] lists (e.g., {'16:9': [1024, 576]}).
        device (str): Device to create the tensor on ('cpu' or 'cuda').

    Returns:
        torch.Tensor: A tensor containing the default height and width (shape [1, 2]).
    """
    # Parse the input aspect ratio
    try:
        width, height = map(float, ar_input.split(':'))
        input_ratio = width / height
    except ValueError:
        raise ValueError("ar_input must be in the format 'width:height' (e.g., '16:9')")
    # Find the closest matching ratio
    closest_ratio = min(
        ratios.keys(), key=lambda ratio: abs(float(ratio) - input_ratio)
    )
    # Get the default height and width for the closest ratio
    default_hw = ratios[closest_ratio]
    # Convert to PyTorch tensor and add batch dimension
    default_hw_tensor = torch.tensor(default_hw, device=device).float().unsqueeze(0)  # Shape: [1, 2]
    return default_hw_tensor

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
        self.sample_steps = args.num_sample_steps # remove hard coding
        self.args = args
        self.ensemble_size = args.ensemble_size
        self.num_sampling_steps = args.num_sampling_steps
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.vae = self.initialize_model()
        save_dir = "output/null_embedding"
        self.null_caption_token, self.null_caption_embs = load_null_caption_embeddings(save_dir)
        self.base_ratios = getattr(ds_utils, f'ASPECT_RATIO_{args.image_size}', ds_utils.ASPECT_RATIO_1024)
        #TODO have tokens extracted
     
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

    def single_infer(self, rgb_batch, cfg_scale=None):
        #TODO change this whole
        """Run inference to generate depth maps for a batch of images."""
        device = self.device

        # Map input images to latent space and normalize latents
        with torch.no_grad():
            rgb_input_latent = self.vae.encode(rgb_batch).latent_dist.sample().mul_(self.vae.config.scaling_factor)
        
        batch_size = rgb_input_latent.shape[0]

        emb_masks = self.null_caption_token.attention_mask
               
        
        caption_embs = self.null_caption_embs
        caption_embs = caption_embs[:, None]
        null_y = self.null_caption_embs.repeat(len(batch_size), 1, 1)[:, None]
        print('Finished embedding')

        model_kwargs = {
            #TODO get data_info as argument into sigle_infer func
                'data_info': {'img_hw': hw_tensor, 'aspect_ratio': torch.tensor([1.0], device=device).unsqueeze(0)}, #TODO Add proper aspect ratios for inferencing
                'mask': emb_masks,
                'input_latent': rgb_input_latent
            }
        with torch.no_grad():
            if args.sampling_function == 'dpm-solver':
                z = torch.randn_like(rgb_input_latent, device=device)
                dpm_solver = DPMS(
                    self.model.forward_with_dpmsolver,
                    condition=caption_embs, #[1,1,300,4096]
                    uncondition=null_y,
                    cfg_scale=cfg_scale,
                    model_kwargs=model_kwargs
                )
                samples = dpm_solver.sample(
                    noise,
                    steps=self.sample_steps,
                    order=2,
                    skip_type="time_uniform",
                    method="multistep",
                )


        #RESIZE #TODO take it to pipe function 
        # hw_str = f"{int(hw[0])}:{int(hw[1])}"  # Convert to 'height:width' string
        # hw_tensor = get_default_hw(hw_str, self.base_ratios, device=device)
        # hw_tensor = torch.Tensor([[512., 512.]]).to("cuda:0")
        
        # # Assuming latent_size is defined elsewhere based on image_size
        # latent_size_h = int(hw_tensor[0, 0].item() // 8)
        # latent_size_w = int(hw_tensor[0, 1].item() // 8)

        # resized_image = resize_to_hw(image_tensor, hw_tensor, device)
        # resized_image = resized_image.unsqueeze(0)


        # Create sampling noise and classifier-free guidance
        
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
        #TODO add resizing according to aspect ratio
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
        
        import time

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
        #TODO Change this
        """Initialize the model, diffusion, and VAE."""
        device = self.device

        latent_size = self.args.image_size // 8
        #For sigma models [512] For [1024, 2048], pe_interpolation will be 2 and 4 respectively
        model = PixArtMS_XL_2(
            input_size=latent_size,
            pe_interpolation=self.args.image_size/512, #image_size / 512
            micro_condition=False,
            model_max_length=300,
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
        if 'pos_embed' in state_dict['state_dict']:
            del state_dict['state_dict']['pos_embed']
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        # Initialize diffusion and VAE
        # diffusion = create_diffusion(str(self.args.num_sampling_steps))
        # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{self.args.vae}").to(device)
        # PIxart sigma vae 
        vae = AutoencoderKL.from_pretrained(f"{args.pipeline_load_from}/vae").to(device)

        logger.info("Model, diffusion, and VAE have been initialized.")
        if self.args.fp16:
            model = model.to(device).half()
            vae = vae.to(device).half()

        return model, vae

def resize_to_hw(image_tensor, hw_tensor, device='cpu'):
    """
    Resizes the input image tensor to the dimensions specified by `hw_tensor`.
    
    Args:
        image_tensor (torch.Tensor): Input image tensor of shape [C, H, W].
        hw_tensor (torch.Tensor): Tensor containing [height, width] of shape [1, 2].
        device (str): Device to move the resized tensor ('cpu' or 'cuda').
        
    Returns:
        torch.Tensor: Resized image tensor with shape [C, new_H, new_W].
    """
    # Extract height and width from hw_tensor
    new_height, new_width = int(hw_tensor[0, 0].item()), int(hw_tensor[0, 1].item())

    print(f"Resizing to new height: {new_height}, new width: {new_width}")

    # Resize the input image tensor using bilinear interpolation
    resized_image = F.interpolate(
        image_tensor.unsqueeze(0),  # Add batch dimension: [1, C, H, W]
        size=(new_height, new_width),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)  # Remove batch dimension: [C, new_H, new_W]

    return resized_image.to(device)


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

            # Main inference
            rgb_int = batch["rgb_int"].squeeze().numpy().astype(np.uint8)  # [3, H, W]
            rgb_int = np.moveaxis(rgb_int, 0, -1)  # [H, W, 3]
            input_image = Image.fromarray(rgb_int)
            depth_pred, colored_depth, pred_uncert = depth_inference.pipe(input_image)
            # depth_pred, colored_depth, pred_uncert = None, None, None
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
    # parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument(
        "--pipeline_load_from", default='/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/depth_estimation_experiments/PixArt-sigma/output/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers',
        type=str, help="Download for loading text_encoder, "
                       "tokenizer and vae from https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"
    )

    parser.add_argument("--image-size", type=int, choices=[256, 512], default=512)
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
    parser.add_argument("--sampling_function", type=str, choices=["ddim", "ddpm", "dpm-solver"], default="ddim", help="Scheduler type to use for inference.")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision for inference.")
    args = parser.parse_args()
    process_image(args)


"""
python infer.py \
--batch-size 10 \
--num-sampling-steps 50 \
--ensemble-size 10 \
--dataset-config config/dataset/data_nyu_test.yaml \
--base-data-dir /mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/depth_estimation_experiments/Marigold/eval_dataset \
--ckpt /mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/depth_estimation_experiments/DiT/PixArt-sigma/output/depth_mixed_training/checkpoints/epoch_3_step_14000.pth \
--output-dir /mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/depth_estimation_experiments/DiT/results/batch_eval/nyu_test/prediction
"""
