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
    _weight *= 0.5  # half the activation magnitude
    _n_proj_out_channel = model.x_embedder.proj.out_channels  # new proj channel
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
    return model

def chw2hwc(chw):
    assert 3 == len(chw.shape)
    if isinstance(chw, torch.Tensor):
        hwc = torch.permute(chw, (1, 2, 0))
    elif isinstance(chw, np.ndarray):
        hwc = np.moveaxis(chw, 0, -1)
    return hwc


def colorize_depth_maps(depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None):
    """Colorize depth maps"""
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()

    # Reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # Colorize depth map
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # Value from 0 to 1
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
    """Decode depth latent into depth map"""
    # Scale latent
    depth_latent = depth_latent / 0.18215
    z = vae.post_quant_conv(depth_latent)
    stacked = vae.decoder(z)
    depth_mean = stacked.mean(dim=1, keepdim=True)
    return depth_mean


def get_rgb_norm(image_path):
    """Normalize the RGB image"""
    image = Image.open(image_path)
    # Resize the image to 512x512
    image = image.resize((512, 512))
    rgb = np.asarray(image.convert("RGB"))
    rgb = np.transpose(rgb, (2, 0, 1)).astype(int)  # [rgb, H, W]
    rgb_norm = rgb / 255.0 * 2.0 - 1.0  # [0, 255] -> [-1, 1]
    rgb_norm = torch.from_numpy(rgb_norm).float().unsqueeze(0)  # [1, C, H, W]
    return rgb_norm




import os
from PIL import Image
import torch
import numpy as np

# ----------------- Load Image(s) Function -----------------
def load_images(path, device):
    """
    Loads a single image or all images from a directory.
    Args:
        path (str): Path to the image file or directory.
        device (str): Device to load the image(s) onto (CPU/GPU).
    Returns:
        tuple: A tuple containing:
            - list of tuples: (processed image tensor, filename)
            - list of tuples: (original image shape, filename)
    """
    image_tensors = []
    original_shapes = []
    
    if os.path.isdir(path):
        # If path is a directory, load all images from the directory
        for filename in os.listdir(path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(path, filename)
                print(f"Loading image: {image_path}")
                image = Image.open(image_path)
                original_shape = image.size  # (width, height)
                image_tensor = get_rgb_norm(image_path).to(device)
                image_tensors.append((image_tensor, original_shape, filename))
    else:
        # If path is a single image file, load the single image
        image = Image.open(path)
        original_shape = image.size  # (width, height)
        image_tensor = get_rgb_norm(path).to(device)
        filename = os.path.basename(path)
        image_tensors.append((image_tensor, original_shape, filename))

    return image_tensors

# ----------------- Inference Function -----------------
def inference(args, rgb, model, diffusion, vae, original_size):
    """
    Runs the inference pipeline to generate a depth map and save the result.
    Args:
        args (Namespace): Command-line arguments or configuration options.
        rgb (torch.Tensor): Pre-processed RGB image tensor.
        model: Pre-initialized model.
        diffusion: Pre-initialized diffusion model.
        vae: Pre-initialized VAE model.
        output_name (str): Filename to save the output depth map.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Map input image to latent space and normalize latents
    with torch.no_grad():
        rgb_input_latent = vae.encode(rgb).latent_dist.sample().mul_(0.18215)

    # Adjust Model for Depth Input
    if 8 != model.x_embedder.proj.weight.shape[1]:
        model = _replace_patchembed_proj(model)

    # Create Sampling Noise and Classifier-Free Guidance
    noise = torch.randn_like(rgb_input_latent, device=device)
    n = rgb_input_latent.shape[0]
    y = torch.zeros(n, dtype=torch.long).to(device)

    noise = torch.cat([noise, noise], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    rgb_input_latent = torch.cat([rgb_input_latent, rgb_input_latent], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale, input_img=rgb_input_latent)

    # Sample images from the diffusion model
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)

    # Decode the depth from latent space
    depth = decode_depth(samples, vae)

    # Reshape back to the original_size
    from torchvision.transforms.functional import InterpolationMode
    depth = resize(depth, original_size[::-1], interpolation=InterpolationMode.BILINEAR, antialias=True)

    # Clipping the values
    depth = torch.clip(depth, -1.0, 1.0)

    # Normalize and colorize the depth map
    depth_pred = (depth + 1.0) / 2.0
    depth_pred = depth_pred.squeeze().detach().cpu().numpy().clip(0, 1)
    depth_colored = colorize_depth_maps(depth_pred, 0, 1, cmap="Spectral").squeeze()
    depth_colored = (depth_colored * 255).astype(np.uint8)
    depth_colored = np.transpose(depth_colored, (1, 2, 0))

    # Return the colorized depth image
    depth_colored_image = Image.fromarray(depth_colored)
    return depth_colored_image, depth_pred

# ----------------- Model Initialization Function -----------------
def initialize_model(args):
    """
    Initializes the model, diffusion, and VAE based on the provided arguments.
    Args:
        args (Namespace): Command-line arguments or configuration options.
    Returns:
        model, diffusion, vae: Initialized model, diffusion model, and VAE.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    ).to(device)

    # Load checkpoint
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    
    if args.depth_ckpt:
        model = _replace_patchembed_proj(model)
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Initialize diffusion and VAE
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    return model, diffusion, vae

# ----------------- Main Entry Function -----------------
def process_images(args):
    """
    Process a single image or all images in a directory for depth inference.
    Args:
        args (Namespace): Command-line arguments or configuration options.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the model, diffusion, and VAE only once
    model, diffusion, vae = initialize_model(args)

    # Load single image or all images in a directory
    image_tensors = load_images(args.image_path, device)

    # Run inference on each image
    for image_tensor, original_shape, filename in image_tensors:
        output_path = os.path.join(args.output_path, f"{os.path.basename(filename)}_depth.png")
        depth_npy_output_path = os.path.join(args.output_path, f"{os.path.basename(filename)}_depth.npy")
        depth_colored_image, depth_pred = inference(args, image_tensor, model, diffusion, vae, original_shape)
        np.save(depth_npy_output_path, depth_pred)
        print(f"Saved depth prediction as npy file at {depth_npy_output_path}")
        depth_colored_image.save(output_path)
        print(f"Saved depth-colored image at {output_path}")


# ----------------- Main Execution Block -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--depth-ckpt", type=bool, default=True)
    parser.add_argument("--image-path", type=str, required=True, help="Path to the input image or directory.")
    parser.add_argument("--output-path", type=str, required=True, help="Path to the output directory.")
    args = parser.parse_args()
    process_images(args)


# TODOs
# >> TESTING
"""
- Testing the inference with many images: DONE
- Testing the inference with one image: DONE
- The basic resize, fix and test it: DONE
- Test npy version, by matching it with the img: DONE
"""

# >> ADDITIONS
"""
- Also save the npy version: DONE
- Add torchvision resize for pred: DONE
- Multibatch inferencing: NOT STARTED
"""

"""
python inference_depth.py \
  --model DiT-XL/2 \
  --image-size 512 \
  --ckpt /mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/DiT/results/000-DiT-XL-2/checkpoints/0000120.pt \
  --image-path /mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/DiT/data/images/rgb_cam_02_fr0085.png \
  --output-path /mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/DiT/results
"""