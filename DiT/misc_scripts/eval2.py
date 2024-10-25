import cv2
import numpy as np
import os
import matplotlib.cm as cm
import argparse

def process_depth_image(image_path, output_folder, cmap="Spectral"):
    # Load the image using cv2
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Normalize the image to range [0, 1]
    image_normalized = image.astype(np.float32)
    image_normalized = (image_normalized - np.min(image_normalized)) / (np.max(image_normalized) - np.min(image_normalized))

    # Apply the "Spectral" colormap
    cmap = cm.get_cmap(cmap)
    image_colored = cmap(image_normalized)

    # Convert RGBA to RGB and scale to [0, 255]
    image_colored_rgb = (image_colored[:, :, :3] * 255).astype(np.uint8)

    # Construct the output path
    basename = os.path.basename(image_path)
    output_path = os.path.join(output_folder, basename)

    # Save the colorized depth map
    cv2.imwrite(output_path, cv2.cvtColor(image_colored_rgb, cv2.COLOR_RGB2BGR))
    print(f"Colorized depth map saved to {output_path}")

def process_folder(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all images in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            process_depth_image(image_path, output_folder)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Colorize depth maps using Spectral colormap")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the input folder containing depth images")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the output folder to save colorized images")
    args = parser.parse_args()

    # Process the entire folder
    process_folder(args.input_folder, args.output_folder)
