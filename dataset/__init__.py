from dataset.base_depth_dataset import BaseDepthDataset, get_pred_name, DatasetMode  # noqa: F401
from dataset.hypersim_dataset import HypersimDataset
from dataset.depth_transform import get_depth_normalizer
import os
from types import SimpleNamespace

cfg_data_split = {'name': 'hypersim', 'disp_name': 'hypersim_train', 'dir': 'Hypersim/processed/train', 'filenames': 'data_split/hypersim/filename_list_train_filtered_subset.txt', 'resize_to_hw': [480, 640]}

cfg_normalizer = SimpleNamespace(
    type='scale_shift_depth', clip=True, norm_min=-1.0, norm_max=1.0, min_max_quantile=0.02
)

depth_transform = get_depth_normalizer(cfg_normalizer=cfg_normalizer)

kwargs = {'augmentation_args': {'lr_flip_p': 0.5}, 'depth_transform': depth_transform}

dataset = HypersimDataset(
        mode=DatasetMode.TRAIN,
        filename_ls_path="data_split/hypersim/filename_list_train_filtered_subset.txt",
        dataset_dir=os.path.join("/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/Marigold/data", "Hypersim/processed/train"),
        **cfg_data_split,
        **kwargs,
    )
