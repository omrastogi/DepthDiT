from .base_depth_dataset import BaseDepthDataset, DepthFileNameMode
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np

# Closly following https://github.com/city-super/MatrixCity/blob/778775c18d2e30e5684fbecb1e759835e8b0b5aa/scripts/load_data.py#L6
"""
max_depth = 65504/10000 = 6.5504 (Since 65504 is invalid_mask in float16)
min_depth = 0 (since not ment)
Scaled by a factor of 10000
"""
class MatrixCityDataset(BaseDepthDataset):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            min_depth=0, 
            max_depth=6.55, 
            has_filled_depth=False,
            name_mode=DepthFileNameMode.rgb_i_d,
            **kwargs,
        )

    def _read_depth_file(self, rel_path):
        depth_in = self._read_depth(rel_path)
        # # Decode MatrixCity depth
        depth_decoded = depth_in / 10000 # cm -> 100m
        return depth_decoded
    
    def _read_depth(self, img_rel_path):
        image_to_read = os.path.join(self.dataset_dir, img_rel_path)
        image = cv2.imread(image_to_read, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[...,0] #(H, W)
        return image

