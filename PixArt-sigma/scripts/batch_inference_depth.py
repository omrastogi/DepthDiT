import os
import sys
import argparse
import logging
import warnings
from pathlib import Path
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from tqdm import tqdm
from omegaconf import OmegaConf
from diffusers.models import AutoencoderKL

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))  # Ensure this points to the correct parent directory

from src.utils.embedding_utils import save_null_caption_embeddings, load_null_caption_embeddings  # noqa: E402
from src.utils.image_utils import chw2hwc, colorize_depth_maps, decode_depth
from src.dataset.base_depth_dataset import BaseDepthDataset, get_pred_name, DatasetMode
from src.dataset import get_dataset
from diffusion import IDDPM, DPMS, SASolverSampler
from diffusion.model.nets import PixArtMS_XL_2, PixArt_XL_2
import diffusion.data.datasets.utils as ds_utils
from tools.download import find_model

warnings.filterwarnings("ignore")  # ignore warning
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def set_env(seed=0):
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    for _ in range(30):
        torch.randn(1, 4, args.image_size, args.image_size)

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

    # print(f"Resizing to new height: {new_height}, new width: {new_width}")

    # Resize the input image tensor using bilinear interpolation
    resized_image = F.interpolate(
        image_tensor.unsqueeze(0),  # Add batch dimension: [1, C, H, W]
        size=(new_height, new_width),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)  # Remove batch dimension: [C, new_H, new_W]
    return resized_image.to(device)

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

def pipe(image_tensor, hw, ensemble_size=10, batch_size=5, device='cpu'):
    """
    Processes a single image through the inference pipeline.

    Args:
        image_tensor (torch.Tensor): The image tensor of shape [C, H, W].
        hw (tuple): The (height, width) of the image.
        batch_size (int): Batch size for processing.
        device (str): Device to perform computations on.

    Returns:
        depth_pred (np.ndarray): The predicted depth map.
        depth_colored_hwc (np.ndarray): The colorized depth map in HWC format.
    """
    # Prepare the hw tensor
    hw_str = f"{int(hw[0])}:{int(hw[1])}"  # Convert to 'height:width' string
    hw_tensor = get_default_hw(hw_str, base_ratios, device=device)

    # Calculate latent sizes
    latent_size_h = int(hw_tensor[0, 0].item() // 8)
    latent_size_w = int(hw_tensor[0, 1].item() // 8)

    # Resize the input image
    resized_image = resize_to_hw(image_tensor, hw_tensor, device)
    resized_image = resized_image.unsqueeze(0).to(weight_dtype)

    # Duplicate the image for ensembling
    duplicated_rgb = resized_image.expand(ensemble_size, -1, -1, -1)
    single_rgb_dataset = TensorDataset(duplicated_rgb)
    single_rgb_loader = DataLoader(
        single_rgb_dataset, batch_size=batch_size, shuffle=False
    )

    depth_pred_ls = []
    for batch in single_rgb_loader:
        resized_image_batch = batch[0]
        # Call single_infer with the batch
        depth_pred = single_infer(
            resized_image_batch, latent_size_h, latent_size_w, batch_size, device
        )
        depth_pred_ls.append(depth_pred.detach())

    # Ensemble the depth predictions
    depth_preds = torch.cat(depth_pred_ls, dim=0)  # Shape: [batch_size, 1, H, W]
    depth_pred = depth_preds.mean(dim=0, keepdim=True)  # Shape: [1, 1, H, W]

    depth_pred = F.interpolate(
        depth_pred.float(), size=hw, mode='bilinear', align_corners=False
    )
    # Post-processing
    depth_pred = depth_pred.squeeze().cpu().numpy()  # Shape: [H, W]
    depth_pred = depth_pred.clip(0, 1)

    # Colorize depth maps using a colormap
    depth_colored = colorize_depth_maps(depth_pred, 0, 1, cmap="Spectral")

    # Convert to uint8 and HWC format
    depth_colored = (depth_colored * 255).astype(np.uint8).squeeze()
    depth_colored_hwc = chw2hwc(depth_colored)
    return depth_pred, depth_colored_hwc

@torch.inference_mode()
def single_infer(resized_image_batch, latent_size_h, latent_size_w, batch_size, device='cpu'):
    """
    Performs inference on a batch of images and returns the depth predictions.

    Args:
        resized_image_batch (torch.Tensor): The resized image batch tensor of shape [batch_size, C, H, W].
        latent_size_h (int): The height of the latent representation.
        latent_size_w (int): The width of the latent representation.
        batch_size (int): Batch size.
        device (str): Device to perform computations on.

    Returns:
        depth_pred (torch.Tensor): The predicted depth maps.
    """
    # Map input images to latent space and normalize latents
    input_latent = vae.encode(resized_image_batch).latent_dist.sample().mul_(vae.config.scaling_factor)

    # Prepare embeddings and masks
    emb_masks = null_caption_token.attention_mask.repeat(batch_size, 1)
    caption_embs = null_caption_embs.repeat(batch_size, 1, 1)[:, None]
    null_y = null_caption_embs.repeat(batch_size, 1, 1)[:, None]

    cfg_scale = args.cfg_scale

    # Prepare model kwargs
    model_kwargs = {
        'data_info': None,
        'mask': emb_masks,
        'input_latent': input_latent
    }

    # Perform sampling
    n = batch_size
    if args.sampling_algo == 'iddpm':
        z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device).repeat(2, 1, 1, 1)
        model_kwargs.update({
            'y': torch.cat([caption_embs, null_y]),
            'cfg_scale': cfg_scale,
            'input_latent': torch.cat([input_latent, input_latent], dim=0)
        })
        diffusion = IDDPM(str(sample_steps))
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
            device=device
        )
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

    elif args.sampling_algo == 'dpm-solver':
        z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device)
        dpm_solver = DPMS(
            model.forward_with_dpmsolver,
            condition=caption_embs,
            uncondition=null_y,
            cfg_scale=cfg_scale,
            model_kwargs=model_kwargs
        )
        samples = dpm_solver.sample(
            z,
            steps=sample_steps,
            order=2,
            skip_type="time_uniform",
            method="multistep",
        )

    elif args.sampling_algo == 'sa-solver':
        sa_solver = SASolverSampler(
            model.forward_with_dpmsolver,
            device=device
        )
        samples = sa_solver.sample(
            S=sample_steps,
            batch_size=n,
            shape=(4, latent_size_h, latent_size_w),
            eta=1,
            conditioning=caption_embs,
            unconditional_conditioning=null_y,
            cfg_scale=cfg_scale,
            model_kwargs=model_kwargs,
        )[0]

    # Decode the samples to get the depth map
    samples = samples.to(weight_dtype)
    depth = decode_depth(samples, vae)

    # Normalize depth values between -1 and 1 to 0 and 1
    depth_pred = (depth + 1.0) / 2.0  # Shape: [batch_size, 1, H, W]

    return depth_pred  # Return tensor without squeezing

def process_image(args):
    """Process images and save depth predictions."""
    # depth_inference = DepthInference(args)
    cfg_data = OmegaConf.load(args.config_path)
    device = "cuda"
    dataset: BaseDepthDataset = get_dataset(
        cfg_data, base_data_dir=args.base_data_dir, mode=DatasetMode.RGB_ONLY
    )
    transform = transforms.ToTensor()
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    with torch.no_grad():
        for batch in tqdm(
            dataloader, desc=f"Inferencing on {dataset.disp_name}", leave=True
        ):

            # Main inference
            rgb_int = batch["rgb_int"].squeeze().numpy().astype(np.uint8)  # [3, H, W]
            rgb_int = np.moveaxis(rgb_int, 0, -1)  # [H, W, 3]
            input_image = Image.fromarray(rgb_int)
            original_shape = input_image.size  # (width, height)
            
                # Convert PIL Image to PyTorch Tensor
            image_tensor = transform(input_image).to(device)  # Shape: [C, H, W]
                
                # Extract height and width
            hw = (original_shape[1], original_shape[0])  # (height, width)
            depth_pred, colored_depth = pipe(image_tensor, hw, ensemble_size=args.ensemble_size, batch_size=args.batch_size, device=device)
            # depth_pred, colored_depth = single_infer(image_tensor, hw, device=device)

            # depth_pred, colored_depth, pred_uncert = depth_inference.pipe(input_image)
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
            # if os.path.exists(save_to):
                # logger.warning(f"Existing file: '{save_to}' will be overwritten.")
            np.save(save_to, depth_pred)

            # Save colored depth as PNG
            colored_depth_basename = get_pred_name(
                rgb_basename, dataset.name_mode, suffix=".png"
            )
            colored_depth_save_to = os.path.join(scene_dir, colored_depth_basename)
            Image.fromarray(colored_depth).save(colored_depth_save_to)
            logger.info(f"Saved colored depth to {colored_depth_save_to} ")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', default=1024, type=int)
    parser.add_argument('--version', default='sigma', type=str)
    parser.add_argument(
        "--pipeline_load_from", default='output/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers',
        type=str, help="Download for loading text_encoder, "
                       "tokenizer and vae from https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"
    )
    parser.add_argument('--model_path', default='output/pretrained_models/PixArt-XL-2-1024x1024.pth', type=str)
    parser.add_argument('--sdvae', action='store_true', help='sd vae')
    parser.add_argument('--cfg_scale', default=4.5, type=float)
    parser.add_argument('--sampling_algo', default='dpm-solver', type=str, choices=['iddpm', 'dpm-solver', 'sa-solver'])
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--step', default=-1, type=int)
    parser.add_argument('--output_dir',default="/data/om/depth-estimation-dit/PixArt-sigma/output", type=str )
    parser.add_argument('--is_depth', action='store_true')
    parser.add_argument("--base_data_dir", type=str, default='/data/om/data/eval_dataset', required=False, help="Path to base data directory.")
    parser.add_argument('--ensemble_size', default=10, type=int, help="Size of the ensemble for predictions.")
    parser.add_argument('--batch_size', default=10, type=int, help="Batch size for processing images.")
    parser.add_argument('--config_path', default='configs/dataset/data_nyu_test.yaml', type=str, help="Path to the configuration file.")
    return parser.parse_args()

def check_and_save_config_summary(args):
    """
    Checks if the configuration summary already exists and validates it against the current arguments.
    If not present, saves the current configuration to a new file.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
        config_summary_path (str): Path to save the configuration summary.

    Raises:
        AssertionError: If an existing configuration summary has mismatched values.
    """
    config_summary_path = os.path.join(args.output_dir, "inference_config_summary.txt")
    # If configuration file exists, validate existing values
    if os.path.exists(config_summary_path):
        with open(config_summary_path, 'r') as existing_config_file:
            existing_config = existing_config_file.read()

        # Extract values of each parameter in existing config
        existing_values = {
            "Model Path": args.model_path,
            "CFG Scale": args.cfg_scale,
            "Sampling Algorithm": args.sampling_algo,
            "Steps": args.step if args.step != -1 else 'default',
            "Batch Size": args.batch_size,
            "Ensemble Size": args.ensemble_size,
            "Image Size": args.image_size,
        }

        # Check each line for consistency with the current args
        for line in existing_config.strip().splitlines():
            key, value = line.split(": ", 1)
            expected_value = str(existing_values.get(key.strip()))

            # Raise an error if there's a mismatch
            assert value.strip() == expected_value, (
                f"Configuration mismatch in '{config_summary_path}' for '{key}': "
                f"expected '{expected_value}', found '{value.strip()}'. Please verify the output directory."
            )

        print("Existing configuration matches current parameters.")

    # If no config file exists, save the current configuration
    else:
        os.makedirs(os.path.dirname(config_summary_path), exist_ok=True)
        with open(config_summary_path, 'w') as config_file:
            config_file.write(f"Model Path: {args.model_path}\n")
            config_file.write(f"CFG Scale: {args.cfg_scale}\n")
            config_file.write(f"Sampling Algorithm: {args.sampling_algo}\n")
            config_file.write(f"Steps: {args.step if args.step != -1 else 'default'}\n")
            config_file.write(f"Batch Size: {args.batch_size}\n")
            config_file.write(f"Ensemble Size: {args.ensemble_size}\n")
            config_file.write(f"Image Size: {args.image_size}\n")
        print(f"Inference configuration saved to {config_summary_path}")


if __name__ == '__main__':
    args = get_args()

    # Setup PyTorch environment
    set_env(args.seed)
    # Save inference configuration summary to output_dir
    check_and_save_config_summary(args)

    # Determine device and data type
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16 if device.type == 'cuda' else torch.float32

    print(f"Using device: {device} with dtype: {weight_dtype}")
    assert args.sampling_algo in ['iddpm', 'dpm-solver', 'sa-solver']

    # Check for saved embeddings, or save them if not present
    save_dir = "output/null_embedding"
    null_caption_token_path = os.path.join(save_dir, "null_caption_token.pt")
    null_caption_embs_path = os.path.join(save_dir, "null_caption_embs.pt")
    if not (os.path.exists(null_caption_token_path) and os.path.exists(null_caption_embs_path)):
        save_null_caption_embeddings(args.pipeline_load_from)

    # Load saved embeddings
    null_caption_token, null_caption_embs = load_null_caption_embeddings(save_dir)
    null_caption_token = null_caption_token.to(device)
    null_caption_embs = null_caption_embs.to(device)

    # Set latent size and model parameters
    latent_size = args.image_size // 8
    max_sequence_length = {"alpha": 120, "sigma": 300}[args.version]
    pe_interpolation = args.image_size / 512
    micro_condition = (args.version == 'alpha' and args.image_size == 1024)
    sample_steps = args.step if args.step != -1 else {
        'iddpm': 100,
        'dpm-solver': 20,
        'sa-solver': 25
    }[args.sampling_algo]

    # Initialize the model with appropriate configuration
    if args.image_size in [512, 1024, 2048] or args.version == 'sigma':
        model = PixArtMS_XL_2(
            input_size=latent_size,
            pe_interpolation=pe_interpolation,
            micro_condition=micro_condition,
            model_max_length=max_sequence_length
        ).to(device, dtype=weight_dtype)
    else:
        model = PixArt_XL_2(
            input_size=latent_size,
            pe_interpolation=pe_interpolation,
            model_max_length=max_sequence_length
        ).to(device, dtype=weight_dtype)

    print(f"Generating sample from checkpoint: {args.model_path}")
    state_dict = find_model(args.model_path)['state_dict']
    state_dict.pop('pos_embed', None)

    # Load the state_dict and modify the model if needed
    if args.is_depth:
        model = _replace_patchembed_proj(model)  # Adjust for depth
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if not args.is_depth:
        model = _replace_patchembed_proj(model)

    print(f'Missing keys: {missing}')
    print(f'Unexpected keys: {unexpected}')

    model.eval()  # Set model to evaluation mode

    # Load the VAE model with appropriate dtype
    vae_path = "output/pretrained_models/sd-vae-ft-ema" if args.sdvae else os.path.join(args.pipeline_load_from, "vae")
    vae = AutoencoderKL.from_pretrained(vae_path).to(device, dtype=weight_dtype)
    print(f"Loaded VAE from: {vae_path}")

    # Load aspect ratio settings
    base_ratios = getattr(ds_utils, f'ASPECT_RATIO_{args.image_size}', ds_utils.ASPECT_RATIO_1024)

    # Process images and save the generated outputs
    process_image(args)
    



"""
python scripts/batch_inference_depth.py \
    --model_path /data/om/models/depth_512_mixed_training/checkpoints/epoch_11_step_52000.pth \
    --base_data_dir /data/om/data/eval_dataset \
    --config_path configs/dataset/data_nyu_test.yaml \
    --output_dir /data/om/depth-estimation-dit/PixArt-sigma/output \
    --sampling_algo dpm-solver \
    --ensemble_size 1 \
    --batch_size 1 \
    --step -1 \
    --is_depth \
"""