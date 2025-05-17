import os
import sys
from pathlib import Path

import numpy as np


current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))  # Ensure this points to the correct parent directory
import warnings
warnings.filterwarnings("ignore")  # ignore warning
from src.utils.embedding_utils import save_null_caption_embeddings, load_null_caption_embeddings
from src.utils.image_utils import chw2hwc, colorize_depth_maps, decode_depth
import re
import argparse
from datetime import datetime
from tqdm import tqdm
import torch
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from transformers import T5EncoderModel, T5Tokenizer
from torch.nn import Conv2d
from torch.nn.parameter import Parameter


from diffusion.model.utils import prepare_prompt_ar
from diffusion import IDDPM, DPMS, SASolverSampler
from tools.download import find_model
from diffusion.model.nets import PixArtMS_XL_2, PixArt_XL_2
from diffusion.data.datasets import get_chunks
import diffusion.data.datasets.utils as ds_utils

import os
from PIL import Image
import torch
from torchvision import transforms


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', default=1024, type=int)
    parser.add_argument('--version', default='sigma', type=str)
    parser.add_argument(
        "--pipeline_load_from", default='/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/depth_estimation_experiments/PixArt-sigma/output/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers',
        type=str, help="Download for loading text_encoder, "
                       "tokenizer and vae from https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"
    )
    parser.add_argument('--txt_file', default='asset/samples.txt', type=str)
    parser.add_argument('--model_path', default='output/pretrained_models/PixArt-XL-2-1024x1024.pth', type=str)
    parser.add_argument('--sdvae', action='store_true', help='sd vae')
    parser.add_argument('--bs', default=1, type=int)
    parser.add_argument('--cfg_scale', default=4.5, type=float)
    parser.add_argument('--sampling_algo', default='dpm-solver', type=str, choices=['iddpm', 'dpm-solver', 'sa-solver'])
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--dataset', default='custom', type=str)
    parser.add_argument('--step', default=-1, type=int)
    parser.add_argument('--save_name', default='test_sample', type=str)
    parser.add_argument('--input_dir',default='/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/depth_estimation_experiments/PixArt-sigma/data/hypersim_vis', type=str )
    parser.add_argument('--is_depth', action='store_true')

    return parser.parse_args()



def load_images(path, device='cpu'):
    """
    Loads a single image or all images from a directory, converts them to tensors, 
    and provides the height and width (hw) of each image.
    
    Args:
        path (str): Path to the image file or directory.
        device (str): Device to load the image(s) onto ('cpu' or 'cuda').
        
    Returns:
        list: A list containing tuples of (image_tensor, hw, filename)
    """
    images = []
    transform = transforms.ToTensor()  # Define the transformation once
    
    if os.path.isdir(path):
        # If path is a directory, load all images from the directory
        for filename in os.listdir(path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(path, filename)
                print(f"Loading image: {image_path}")
                
                try:
                    # Open image and ensure it's in RGB format
                    with Image.open(image_path).convert('RGB') as img:
                        original_shape = img.size  # (width, height)
                        
                        # Convert PIL Image to PyTorch Tensor
                        image_tensor = transform(img).to(device)  # Shape: [C, H, W]
                        
                        # Extract height and width
                        hw = (original_shape[1], original_shape[0])  # (height, width)
                        
                        images.append((image_tensor, hw, filename))
                except Exception as e:
                    print(f"Failed to load {image_path}: {e}")
    else:
        # If path is a single image file, load the single image
        try:
            with Image.open(path).convert('RGB') as img:
                original_shape = img.size  # (width, height)
                
                # Convert PIL Image to PyTorch Tensor
                image_tensor = transform(img).to(device)  # Shape: [C, H, W]
                
                # Extract height and width
                hw = (original_shape[1], original_shape[0])  # (height, width)
                
                filename = os.path.basename(path)
                images.append((image_tensor, hw, filename))
        except Exception as e:
            print(f"Failed to load {path}: {e}")
    
    return images


# Example usage:
# images = load_images('path/to/image_or_directory', device='cuda')
# for img_tensor, hw, filename in images:
#     print(f"{filename}: Tensor shape = {img_tensor.shape}, Height-Width = {hw}")


def set_env(seed=0):
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    for _ in range(30):
        torch.randn(1, 4, args.image_size, args.image_size)

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



import os
from tqdm import tqdm
import torch
from torchvision.utils import save_image

import torch
import torch.nn.functional as F

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


@torch.inference_mode()
def visualize(image_list, sample_steps, cfg_scale, save_root, ratios, device='cpu'):
    """
    Generates and saves images based on the provided image list.
    
    Args:
        image_list (list): List of tuples containing (image_tensor, hw, filename).
        sample_steps (int): Number of sampling steps for the diffusion model.
        cfg_scale (float): Classifier-free guidance scale.
        save_root (str): Directory to save the generated images.
        prepare_hw_func (callable): Function to prepare hw tensor.
        device (str): Device to perform computations on ('cpu' or 'cuda').
        
    Returns:
        None
    """
    bs = 1  # Fixed batch size
    for (image_tensor, hw, filename) in tqdm(image_list, unit='image'):
        save_path = os.path.join(save_root, f"{filename[:100]}.jpg")
        if os.path.exists(save_path):
            print(f"Image already exists: {save_path}, skipping.")
            continue
        
        # Prepare the hw tensor using the new function
        hw_str = f"{int(hw[0])}:{int(hw[1])}"  # Convert to 'height:width' string
        hw_tensor = get_default_hw(hw_str, ratios, device=device)
    
        
        # Assuming latent_size is defined elsewhere based on image_size
        latent_size_h = int(hw_tensor[0, 0].item() // 8)
        latent_size_w = int(hw_tensor[0, 1].item() // 8)
        

        #TODO Resize the the input image, convert into latent - Test
        resized_image = resize_to_hw(image_tensor, hw_tensor, device)
        resized_image = resized_image.unsqueeze(0).to(weight_dtype)


        # Map input images to latent space and normalize latents
        with torch.no_grad():
            input_latent = vae.encode(resized_image).latent_dist.sample().mul_(vae.config.scaling_factor)


        # Example prompt (since prompt is no longer used, you might need to adjust your model accordingly)
        prompts = [""]  # Placeholder if your model requires prompts
        
        #TODO should I use them as None
        emb_masks = null_caption_token.attention_mask
        # emb_masks = None
        
        
        caption_embs = null_caption_embs
        caption_embs = caption_embs[:, None]
        #TODO just use null_caption_embs P1
        null_y = null_caption_embs.repeat(len(prompts), 1, 1)[:, None]
        print('Finished embedding')
        
        
        with torch.no_grad():
            n = len(prompts)
            #TODO send input latent p0 - to test
            #TODO also make changes in the forward() function - To test 
            model_kwargs = {
                'data_info': {'img_hw': hw_tensor, 'aspect_ratio': torch.tensor([1.0], device=device).unsqueeze(0)},
                'mask': emb_masks,
                'input_latent': input_latent
            }
            
            if args.sampling_algo == 'iddpm':
                # Create sampling noise
                z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device).repeat(2, 1, 1, 1)
                model_kwargs.update({
                    'y': torch.cat([caption_embs, null_y]),
                    'cfg_scale': cfg_scale
                })
                diffusion = IDDPM(str(sample_steps))
                # Sample images
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
                # Create sampling noise
                z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device)
                dpm_solver = DPMS(
                    model.forward_with_dpmsolver,
                    condition=caption_embs, #[1,1,300,4096]
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
                # Create sampling noise
                sa_solver = SASolverSampler(
                    model.forward_with_dpmsolver,
                    device=device
                )
                samples = sa_solver.sample(
                    S=25,
                    batch_size=n,
                    shape=(4, latent_size_h, latent_size_w),
                    eta=1,
                    conditioning=caption_embs,
                    unconditional_conditioning=null_y,
                    unconditional_guidance_scale=cfg_scale,
                    model_kwargs=model_kwargs,
                )[0]
        
        samples = samples.to(weight_dtype)
        depth = decode_depth(samples, vae)
        depth = torch.clip(depth, -1.0, 1.0)  # TODO: Check this step

        # Normalize depth values between 0 and 1
        depth_pred = (depth + 1.0) / 2.0
        depth_pred = depth_pred.squeeze().detach().cpu().numpy()
        depth_pred = depth_pred.clip(0, 1)

        # Colorize depth maps using a colormap
        depth_colored = colorize_depth_maps(depth_pred, 0, 1, cmap="Spectral").squeeze()

        # Convert to uint8 for wandb logging
        depth_colored = (depth_colored * 255).astype(np.uint8)
        depth_colored_hwc = chw2hwc(depth_colored)
        # depth_colored_hwc_tensor = torch.tensor(depth_colored_hwc, dtype=torch.uint8).permute(2, 0, 1).unsqueeze(0)

        torch.cuda.empty_cache()
        
        # Save images
        os.umask(0o000)  # File permission: 666; dir permission: 777
        save_path = os.path.join(save_root, f"{filename[:100]}.jpg")
        print("Saving path:", save_path)
        Image.fromarray(depth_colored_hwc).save(save_path)


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


if __name__ == '__main__':
    args = get_args()
    # Setup PyTorch:
    seed = args.seed
    set_env(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert args.sampling_algo in ['iddpm', 'dpm-solver', 'sa-solver']

    # only support fixed latent size currently
    latent_size = args.image_size // 8
    max_sequence_length = {"alpha": 120, "sigma": 300}[args.version]
    pe_interpolation = args.image_size / 512
    micro_condition = True if args.version == 'alpha' and args.image_size == 1024 else False
    sample_steps_dict = {'iddpm': 100, 'dpm-solver': 20, 'sa-solver': 25}
    sample_steps = args.step if args.step != -1 else sample_steps_dict[args.sampling_algo]
    weight_dtype = torch.float16
    print(f"Inference with {weight_dtype}")

    # model setting
    micro_condition = True if args.version == 'alpha' and args.image_size == 1024 else False
    if args.image_size in [512, 1024, 2048] or args.version == 'sigma':
        model = PixArtMS_XL_2(
            input_size=latent_size,
            pe_interpolation=pe_interpolation,
            micro_condition=micro_condition,
            model_max_length=max_sequence_length,
        ).to(device)
    else:
        model = PixArt_XL_2(
            input_size=latent_size,
            pe_interpolation=pe_interpolation,
            model_max_length=max_sequence_length,
        ).to(device)

    print("Generating sample from ckpt: %s" % args.model_path)
    state_dict = find_model(args.model_path)
    if 'pos_embed' in state_dict['state_dict']:
        del state_dict['state_dict']['pos_embed']
    # Replacing projection layer in PatchEmbed
    # model = _replace_patchembed_proj(model)
    if args.is_depth:
        # The state_dict is already modified for depth, so modify the model first
        model = _replace_patchembed_proj(model)
        missing, unexpected = model.load_state_dict(state_dict['state_dict'], strict=False)
    else:
        # For a vanilla DiT model:
        # Load the state_dict first, and then modify the model afterwards
        missing, unexpected = model.load_state_dict(state_dict['state_dict'], strict=False)
        model = _replace_patchembed_proj(model)

    print('Missing keys: ', missing)
    print('Unexpected keys', unexpected)

    model.eval()
    model.to(weight_dtype)

    # if 8 != model.x_embedder.proj.weight.shape[1]:
    #     model = _replace_patchembed_proj(model)

    # How do they decide the ratio
    base_ratios = getattr(ds_utils, f'ASPECT_RATIO_{args.image_size}', ds_utils.ASPECT_RATIO_1024)

    if args.sdvae:
        # pixart-alpha vae link: https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/sd-vae-ft-ema
        vae = AutoencoderKL.from_pretrained("output/pretrained_models/sd-vae-ft-ema").to(device).to(weight_dtype)
    else:
        # pixart-Sigma vae link: https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers/tree/main/vae
        vae = AutoencoderKL.from_pretrained(f"{args.pipeline_load_from}/vae").to(device).to(weight_dtype)

    # tokenizer = T5Tokenizer.from_pretrained(args.pipeline_load_from, subfolder="tokenizer")
    # text_encoder = T5EncoderModel.from_pretrained(args.pipeline_load_from, subfolder="text_encoder").to(device)

    # null_caption_token = tokenizer("", max_length=max_sequence_length, padding="max_length", truncation=True, return_tensors="pt").to(device)
    # null_caption_embs = text_encoder(null_caption_token.input_ids, attention_mask=null_caption_token.attention_mask)[0]

    # Check if the .pt files exist, otherwise save them
    save_dir = "output/null_embedding"
    # if not (os.path.exists(os.path.join(save_dir, "null_caption_token.pt")) and
    #         os.path.exists(os.path.join(save_dir, "null_caption_embs.pt"))):
    #     save_null_caption_embeddings(args.pipeline_load_from, accelerator)

    # Load the saved embeddings and tokens
    null_caption_token, null_caption_embs = load_null_caption_embeddings(save_dir)

    work_dir = os.path.join(*args.model_path.split('/')[:-2])
    work_dir = '/'+work_dir if args.model_path[0] == '/' else work_dir

    # data setting
    with open(args.txt_file, 'r') as f:
        items = [item.strip() for item in f.readlines()]

    # empty_items = ["" for item in items]

    # img save setting
    try:
        epoch_name = re.search(r'.*epoch_(\d+).*', args.model_path).group(1)
        step_name = re.search(r'.*step_(\d+).*', args.model_path).group(1)
    except:
        epoch_name = 'unknown'
        step_name = 'unknown'

    output_dir = os.path.join(work_dir, 'output_dir')
    image_list = []

    for image_name in os.listdir(args.input_dir):
        image_path = os.path.join(args.input_dir, image_name)
        output = load_images(image_path, device=device)
        image_list.append(output[0])
        
    img_save_dir = os.path.join(work_dir, 'vis')
    os.umask(0o000)  # file permission: 666; dir permission: 777
    os.makedirs(img_save_dir, exist_ok=True)

    save_root = os.path.join(img_save_dir, f"{datetime.now().date()}_{args.dataset}_epoch{epoch_name}_step{step_name}_scale{args.cfg_scale}_step{sample_steps}_size{args.image_size}_bs{args.bs}_samp{args.sampling_algo}_seed{seed}")
    os.makedirs(save_root, exist_ok=True)
    # visualize(items, args.bs, sample_steps, args.cfg_scale, image_list)
    visualize(image_list, sample_steps, args.cfg_scale, save_root, base_ratios, device=device)


