import os
from PIL import Image
import torch
from infer import DepthInference
import argparse
import numpy as np

def load_images(path):
    """
    Loads a single image or all images from a directory.
    Args:
        path (str): Path to the image file or directory.
        device (str): Device to load the image(s) onto (CPU/GPU).
    Returns:
        list: A list containing tuples of (image, original shape, filename)
    """
    images = []
    
    if os.path.isdir(path):
        # If path is a directory, load all images from the directory
        for filename in os.listdir(path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(path, filename)
                print(f"Loading image: {image_path}")
                image = Image.open(image_path)
                original_shape = image.size  # (width, height)
                images.append((image, original_shape, filename))
    else:
        # If path is a single image file, load the single image
        image = Image.open(path)
        original_shape = image.size  # (width, height)
        filename = os.path.basename(path)
        images.append((image, original_shape, filename))

    return images

def process_images(args):
    """
    Process images for depth inference in batches.
    Args:
        args (Namespace): Command-line arguments or configuration options.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    depth_inference = DepthInference(args)
    # Load images using the provided load_images function
    image_data_list = load_images(args.image_path)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    for image_data in image_data_list:
        # Unpack the image data
        input_image, original_shape, filename = image_data
        # Convert image to tensor
        # Run inference on the image
        depth_pred, colored_depth, pred_uncert = depth_inference.pipe(input_image)
        # Save results for the image
        output_path = os.path.join(args.output_dir, f"{os.path.basename(filename)}_depth.png")
        depth_npy_output_path = os.path.join(args.output_dir, f"{os.path.basename(filename)}_depth.npy")
        colored_depth.save(output_path)
        print(f"Saved depth-colored image at {output_path}")
        if args.save_npy:
            np.save(depth_npy_output_path, depth_pred)
            print(f"Saved depth prediction as npy file at {depth_npy_output_path}")




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
    process_images(args)


"""
python inference_depth2.py \
--batch-size 10 \
--num-sampling-steps 50 \
--ensemble-size 10 \
--fp16 \
--image-path /mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/depth_estimation_experiments/DiT/data/hypersim_vis \
--ckpt /mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/depth_estimation_experiments/DiT/results/model_vkitti_hypersim_4_epoch_multires/checkpoints/0014000.pt \
--output-dir /mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/depth_estimation_experiments/DiT/model_vkitti_hypersim_4_epoch_multires_vis

"""