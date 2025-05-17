# %%
import os
import cv2
import numpy as np
import torch.nn as nn
from urllib.request import urlretrieve
from PIL import Image

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

def load_depth_from_tar(file_data, is_float16=True):
    file_array = np.frombuffer(file_data, dtype=np.uint8)
    image = cv2.imdecode(file_array, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if is_float16:
        invalid_mask = (image == 65504)  # Detect invalid depth values for float16
    else:
        invalid_mask = None
    image = image / 10000  # Convert depth values from cm to meters
    return image, invalid_mask

def load_depth(depth_path, is_float16=True):
    image = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[...,0] #(H, W)
    if is_float16==True:
        invalid_mask=(image==65504)
    else:
        invalid_mask=None
    image = image / 10000 # cm -> 100m
    return image, invalid_mask


def convert_tar_path(input_path):
    directory, file_name = os.path.split(input_path)
    return directory.replace("big_city", "big_city_depth")

def convert_tar_path_to_sampled(input_path):
    directory, file_name = os.path.split(input_path)
    return directory.replace("big_city", "sampled_big_city")

def convert_image_path(input_path):
    directory, file_name = os.path.split(input_path)
    new_directory = directory + "_depth"
    base_name, ext = os.path.splitext(file_name)
    new_file_name = f"{base_name}.exr" if ext == ".png" else file_name
    file_path = os.path.join(new_directory, new_file_name)
    if "./" in file_path:
        file_path = file_path.replace("./", "")
    return file_path
#----------------------------------------------------------------------------
#%%
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import time
from IPython.display import clear_output
import pandas as pd 
# Use inline plotting
%matplotlib inline
df = pd.read_csv('csv/sampled_data.csv')
base_path = "/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/depth_estimation_experiments/DiT/depth_datasets/MatrixCity/MatrixCity"

for i, row in df.iterrows():
    tar_file = os.path.join(base_path, row["Tar File"]) if not os.path.isabs(row["Tar File"]) else row["Tar File"]
    file_path = os.path.join(convert_tar_path(tar_file), convert_image_path(row["File Path Inside Tar"]))
    rgb_file_path = os.path.join(convert_tar_path_to_sampled(tar_file), row["File Path Inside Tar"].lstrip("./"))
    depth_image, invalid_mask = load_depth(file_path)
    depth_image_real = depth_image.copy()

    # Logarithmic depth values
    depth_image_log = np.log(depth_image)
    depth_image_new = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())
    cmap = cmap = cm.Spectral
    depth_image_clipped = np.clip(depth_image_new, 0, 1)
    spectral_image = cmap(depth_image_clipped)[:, :, :3]  # Remove alpha channel

    # Depth values
    depth_image_new_log = (depth_image_log - depth_image_log.min()) / (depth_image_log.max() - depth_image_log.min())
    cmap = cm.Spectral
    depth_image_clipped_log = np.clip(depth_image_new_log, 0, 1)
    spectral_image_log = cmap(depth_image_clipped_log)[:, :, :3]  # Remove alpha channel

    rgb_image = plt.imread(rgb_file_path)

    # Clear the previous output
    clear_output(wait=True)

    # Create a new figure
    plt.figure(figsize=(24, 8))

    # Plot the RGB image
    plt.subplot(1, 3, 1)
    plt.imshow(rgb_image)
    plt.title("RGB Image")
    plt.axis('off')  # Hide axes for better visualization

    # Plot the spectral image
    plt.subplot(1, 3, 2)
    plt.imshow(spectral_image)
    plt.title("Spectral Image")
    plt.axis('off')  # Hide axes for better visualization

    # Plot the spectral image with logarithmic depth
    plt.subplot(1, 3, 3)
    plt.imshow(spectral_image_log)
    plt.title("Spectral Image (Log Depth)")
    plt.axis('off')  # Hide axes for better visualization

    # Add max and min values as text
    max_value = depth_image_real.max()
    min_value = depth_image_real.min()
    plt.text(
        0.05, 0.95, 
        f"Max: {max_value:.2f}\nMin: {min_value:.2f}", 
        color='white', fontsize=12, backgroundcolor='black', 
        verticalalignment='top', horizontalalignment='left', 
        transform=plt.gca().transAxes
    )
    # Save the plots for future reviews
    dir_to_save = "big_city_log_depth_visuals"
    output_path = rgb_file_path.replace("big_city", dir_to_save)
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Save the combined figure
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    
    # Show the plot
    plt.show()
    # Optional: Add a short pause to control update speed
    # time.sleep(2)

    # Close the plot to destroy the figure
    plt.close()
# %%
