import pyrallis
from omegaconf import OmegaConf
from dataclasses import dataclass, field
from typing import Any
import torch
from diffusion.utils.config import SanaConfig
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from accelerate import Accelerator, InitProcessGroupKwargs

from diffusion import DPMS, FlowEuler, Scheduler
from diffusion.data.builder import build_dataloader, build_dataset
from diffusion.data.wids import DistributedRangedSampler
from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder, get_vae, vae_decode, vae_encode
from diffusion.model.respace import compute_density_for_timestep_sampling
from diffusion.utils.checkpoint import load_checkpoint, save_checkpoint
from diffusion.utils.config import SanaConfig
from diffusion.utils.data_sampler import AspectRatioBatchSampler
from diffusion.utils.dist_utils import clip_grad_norm_, flush, get_world_size
from diffusion.utils.logger import LogBuffer, get_root_logger
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.misc import DebugUnderflowOverflow, init_random_seed, read_config, set_random_seed
from diffusion.utils.optimizer import auto_scale_lr, build_optimizer

from src.dataset import BaseDepthDataset, DatasetMode, get_dataset
from src.dataset.depth_transform import get_depth_normalizer
from src.dataset.dist_mixed_sampler import DistributedMixedBatchSampler
from src.dataset.mixed_sampler import MixedBatchSampler
from src.dataset.hypersim_dataset import HypersimDataset
from src.utils.image_utils import decode_depth, colorize_depth_maps, chw2hwc, encode_depth
from src.utils.embedding_utils import load_null_caption_embeddings, save_null_caption_embeddings
from src.utils.multi_res_noise import multi_res_noise_like

def loader_info(loader, name):
    print(f"--- {name} ---")
    print(f"Batch size: {loader.batch_size}")
    print(f"Number of batches: {len(loader)}")
    print(f"Total dataset size: {len(loader.dataset)}")
    print(f"Number of workers: {loader.num_workers}")
    print()


@pyrallis.wrap()
def main(config: SanaConfig):
    # Convert Pyrallis-configured dataclass to OmegaConf DictConfig
    structured_config = OmegaConf.structured(config)

    # Load the dataset configuration using OmegaConf
    dataset_config = OmegaConf.load("configs/depth_dataset.yaml")

    # Convert structured_config to an unstructured DictConfig
    unstructured_config = OmegaConf.to_container(structured_config, resolve=True)
    unstructured_config = OmegaConf.create(unstructured_config)

    # Merge the dataset config into the unstructured config
    merged_config = OmegaConf.merge(unstructured_config, dataset_config)

    print("\nFinal Combined Configuration:")
    print(OmegaConf.to_yaml(merged_config))

    # Pass the merged configuration to your function
    train_loader, val_loader = create_datasets(merged_config, rank=0, world_size=2)
    
    # Test the train_loader
    loader_info(train_loader, "Train Loader")
    loader_info(val_loader, "Validation Loader")


def create_datasets(cfg, rank, world_size):
    if cfg.train.seed is None:
        loader_generator = None
    else:
        loader_generator = torch.Generator().manual_seed(cfg.train.seed)

    # Training dataset
    depth_transform = get_depth_normalizer(
        cfg_normalizer=cfg.depth_normalization
    )

    train_dataset: BaseDepthDataset = get_dataset(
        cfg.dataset.train,
        base_data_dir=cfg.paths.base_data_dir,
        mode=DatasetMode.TRAIN,
        augmentation_args=cfg.augmentation,
        depth_transform=depth_transform,
    )

    if "mixed" == cfg.dataset.train.name:
        dataset_ls = train_dataset
        assert len(cfg.dataset.train.prob_ls) == len(
            dataset_ls
        ), "Lengths don't match: `prob_ls` and `dataset_list`"
        concat_dataset = ConcatDataset(dataset_ls)

        sampler = DistributedMixedBatchSampler(
            src_dataset_ls=dataset_ls,
            batch_size=cfg.train.train_batch_size,
            drop_last=True,
            shuffle=True,
            world_size=world_size,
            rank=rank,
            prob=cfg.dataset.train.prob_ls,
            generator=loader_generator,
        )

        train_loader = DataLoader(
            concat_dataset,
            batch_sampler=sampler,
            num_workers=cfg.train.num_workers,
        )
    else:
        sampler = DistributedSampler(
            train_dataset, 
            num_replicas=world_size, 
            rank=rank, 
            shuffle=True,
            drop_last=True, 
            seed=cfg.train.seed
        )

        train_loader = DataLoader(
            dataset=train_dataset,
            sampler=sampler,
            batch_size=cfg.train.train_batch_size,
            num_workers=cfg.train.num_workers,
            drop_last=True,
            generator=loader_generator,
        )
    
    # For Validation 
    val_dataset: BaseDepthDataset = get_dataset(
            cfg.dataset.val,
            base_data_dir=cfg.paths.base_data_dir,
            mode=DatasetMode.TRAIN,
            depth_transform=depth_transform,
            drop_last=True
        )
    
    if "mixed" == cfg.dataset.val.name:
        val_dataset = ConcatDataset(val_dataset)

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.validation.batch_size,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=False
    )
    return train_loader, val_loader

if __name__ == "__main__":
    main()
