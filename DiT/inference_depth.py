import argparse
import os

import matplotlib
import numpy as np
import torch
from PIL import Image
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL

from src.dataset.base_depth_dataset import BaseDepthDataset, get_pred_name, DatasetMode  # noqa: F401
from src.dataset.depth_transform import get_depth_normalizer
from src.dataset.hypersim_dataset import HypersimDataset
from src.diffusion import create_diffusion
from src.models.download import find_model
from src.models.models import DiT_models

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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
def inference(args, rgb_batch, model, diffusion, vae, original_shapes):
    """
    Runs the inference pipeline to generate depth maps for a batch of images.
    Args:
        args (Namespace): Command-line arguments or configuration options.
        rgb_batch (torch.Tensor): Batch of pre-processed RGB image tensors.
        model: Pre-initialized model.
        diffusion: Pre-initialized diffusion model.
        vae: Pre-initialized VAE model.
        original_shapes (list): List of original image shapes for resizing.
    Returns:
        depth_colored_images (list of PIL Images): Colorized depth maps.
        depth_preds (list of numpy arrays): Depth predictions.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Map input images to latent space and normalize latents
    with torch.no_grad():
        rgb_input_latent = vae.encode(rgb_batch).latent_dist.sample().mul_(0.18215)

    # Create Sampling Noise and Classifier-Free Guidance
    batch_size = rgb_input_latent.shape[0]
    noise = torch.randn_like(rgb_input_latent, device=device)
    y = torch.zeros(batch_size, dtype=torch.long).to(device)

    # Prepare inputs for classifier-free guidance
    noise = torch.cat([noise, noise], 0)
    y_null = torch.tensor([1000] * batch_size, device=device)
    y = torch.cat([y, y_null], 0)
    rgb_input_latent = torch.cat([rgb_input_latent, rgb_input_latent], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale, input_img=rgb_input_latent)

    # Sample images from the diffusion model
    if args.scheduler == "ddim":
        samples = diffusion.ddim_sample_loop(
            model.forward_with_cfg, noise.shape, noise, clip_denoised=False,
            model_kwargs=model_kwargs, progress=True, device=device
        )
    elif args.scheduler == "ddpm":
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, noise.shape, noise, clip_denoised=False,
            model_kwargs=model_kwargs, progress=True, device=device
        )
    samples, _ = samples.chunk(2, dim=0)  # Discard the unconditional samples

    # Decode the depth from latent space
    with torch.no_grad():
        depth = decode_depth(samples, vae)  # Keep only the first channel

    # Initialize lists to store outputs
    depth_colored_images = []
    depth_preds = []

    # Post-process each depth map separately
    for i in range(batch_size):
        # Reshape back to the original size
        resized_depth = torch.nn.functional.interpolate(
            depth[i:i+1], size=original_shapes[i][::-1], mode='bilinear', align_corners=False
        )

        # Clipping the values
        resized_depth = torch.clamp(resized_depth, -1.0, 1.0)

        # Normalize and colorize the depth map
        depth_pred = (resized_depth + 1.0) / 2.0
        depth_pred = depth_pred.squeeze().cpu().numpy().clip(0, 1)

        depth_colored = colorize_depth_maps(depth_pred, 0, 1, cmap="Spectral").squeeze()
        depth_colored = (depth_colored * 255).astype(np.uint8)
        depth_colored = np.transpose(depth_colored, (1, 2, 0))
        depth_colored_image = Image.fromarray(depth_colored)

        # Append to the lists
        depth_colored_images.append(depth_colored_image)
        depth_preds.append(depth_pred)

    return depth_colored_images, depth_preds

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

    # Modify the model architecture before loading state dict
    if 8 != model.x_embedder.proj.weight.shape[1]:
        model = _replace_patchembed_proj(model)

    # Load checkpoint
    checkpoint = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    if "model" in checkpoint:  # supports checkpoints from train.py
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Initialize diffusion and VAE
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    return model, diffusion, vae

# ----------------- Main Entry Function -----------------
def process_images(args):
    """
    Process images for depth inference in batches.
    Args:
        args (Namespace): Command-line arguments or configuration options.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = args.batch_size

    # Initialize the model, diffusion, and VAE only once
    model, diffusion, vae = initialize_model(args)

    # Load images using the provided load_images function
    image_data_list = load_images(args.image_path, device)

    # Helper function to create batches from the image data list
    def batchify(data_list, batch_size):
        for i in range(0, len(data_list), batch_size):
            yield data_list[i:i+batch_size]
    # Check if the output path doesn't exist and create the directory if needed
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    # Process images in batches
    for batch_data in batchify(image_data_list, batch_size):
        # Unpack the batch data
        image_tensors, original_shapes, filenames = zip(*batch_data)
        # Stack image tensors into a batch
        image_batch = torch.cat(image_tensors, dim=0)  # Assuming each tensor has shape [1, C, H, W]
        # Run inference on the batch
        depth_colored_images, depth_preds = inference(args, image_batch, model, diffusion, vae, original_shapes)
        # Save results for each image in the batch
        for i in range(len(filenames)):
            filename = filenames[i]
            depth_colored_image = depth_colored_images[i]
            depth_pred = depth_preds[i]
            output_path = os.path.join(args.output_path, f"{os.path.basename(filename)}_depth.png")
            depth_npy_output_path = os.path.join(args.output_path, f"{os.path.basename(filename)}_depth.npy")
            depth_colored_image.save(output_path)
            print(f"Saved depth-colored image at {output_path}")
            if args.save_npy:
                np.save(depth_npy_output_path, depth_pred)
                print(f"Saved depth prediction as npy file at {depth_npy_output_path}")


# ----------------- Main Execution Block -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--image-path", type=str, required=True, help="Path to the input image or directory.")
    parser.add_argument("--output-path", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for processing images.")
    parser.add_argument("--save-npy", action="store_true", help="Save depth as npy")
    parser.add_argument("--scheduler", type=str, choices=["ddim", "ddpm"], default="ddim", help="Scheduler type to use for inference.")
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
- Multibatch inferencing: DONE
"""

"""
python inference_depth.py \
  --model DiT-XL/2 \
  --image-size 512 \
  --batch-size 10 \
  --num-sampling-steps 50 \
  --ckpt /mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/depth_estimation_experiments/DiT/results/048-DiT-XL-2-training--1015-23:39:58/checkpoints/0014000.pt \
  --image-path /mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/depth_estimation_experiments/DiT/data/lab_img \
  --output-path /mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/depth_estimation_experiments/DiT/results/lab_mixed_training 
"""