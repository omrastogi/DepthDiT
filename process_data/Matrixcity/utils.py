import os
import io
import cv2
import tarfile
import torch
import open_clip
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
