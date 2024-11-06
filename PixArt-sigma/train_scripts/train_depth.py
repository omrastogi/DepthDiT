import argparse
import datetime
import gc
import os
import sys
import time
import types
import warnings
from pathlib import Path
from tqdm import tqdm
import wandb
from copy import deepcopy
from omegaconf import OmegaConf
from collections import OrderedDict
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

import numpy as np
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedType
from diffusers.models import AutoencoderKL
from transformers import T5EncoderModel, T5Tokenizer
from mmcv.runner import LogBuffer
from PIL import Image
from torch.utils.data import RandomSampler
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn import Conv2d
from torch.nn.parameter import Parameter

from diffusion import IDDPM, DPMS
from diffusion.data.builder import build_dataset, build_dataloader, set_data_root
from diffusion.model.builder import build_model
from diffusion.utils.checkpoint import save_checkpoint, load_checkpoint
from diffusion.utils.data_sampler import AspectRatioBatchSampler
from diffusion.utils.dist_utils import synchronize, get_world_size, clip_grad_norm_, flush, get_rank
from diffusion.utils.logger import get_root_logger, rename_file_with_creation_time
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.misc import set_random_seed, read_config, init_random_seed, DebugUnderflowOverflow
from diffusion.utils.optimizer import build_optimizer, auto_scale_lr

from src.dataset import BaseDepthDataset, DatasetMode, get_dataset
from src.dataset.depth_transform import get_depth_normalizer
from src.dataset.dist_mixed_sampler import DistributedMixedBatchSampler
from src.dataset.mixed_sampler import MixedBatchSampler
from src.dataset.hypersim_dataset import HypersimDataset
from src.utils.image_utils import decode_depth, colorize_depth_maps, chw2hwc, encode_depth
from src.utils.embedding_utils import load_null_caption_embeddings, save_null_caption_embeddings
warnings.filterwarnings("ignore")  # ignore warning


def set_fsdp_env():
    os.environ["ACCELERATE_USE_FSDP"] = 'true'
    os.environ["FSDP_AUTO_WRAP_POLICY"] = 'TRANSFORMER_BASED_WRAP'
    os.environ["FSDP_BACKWARD_PREFETCH"] = 'BACKWARD_PRE'
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = 'PixArtBlock'

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Update the EMA model parameters using vectorized operations.
    """
    with torch.no_grad():
        for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
            ema_param.mul_(decay).add_(model_param, alpha=1 - decay)


# def update_ema(ema_model, model, decay=0.9999):
#     """
#     Step the EMA model towards the current model.
#     """
#     import time

#     start_time = time.time()
#     ema_params = OrderedDict(ema_model.named_parameters())
#     model_params = OrderedDict(model.named_parameters())

#     for name, param in model_params.items():
#         # Remove 'module.' prefix if present
#         name_in_ema = name.replace('module.', '')
#         if name_in_ema in ema_params:
#             ema_param = ema_params[name_in_ema]
#             ema_param.mul_(decay).add_(param.data, alpha=1 - decay)
#         else:
#             print(f"Parameter {name_in_ema} not found in EMA model.")

#     end_time = time.time()
#     print(f"EMA update took {end_time - start_time} seconds")

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

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

@torch.inference_mode()
def log_validation(model, loader, vae, device, step):
    torch.cuda.empty_cache()
    model = accelerator.unwrap_model(model).eval()

    print("Validation started")

    # Get the next batch
    batch = next(iter(loader))
    rgb = batch["rgb_norm"].to(device).to(torch.float16)
    rgb_int = batch["rgb_int"].to(device).to(torch.float16)  # Real RGB images from batch

    # Mentioning params
    batch_size = rgb.shape[0]

    wandb_images = []  # Collect all images to log at once

    for i in range(batch_size):
        with torch.no_grad():  # Map input images to latent space + normalize latents:
            rgb_input_latent = (
                vae.encode(rgb[i].unsqueeze(0)).latent_dist.sample() * vae.config.scaling_factor
            )
        
        latent_size_h, latent_size_w = rgb_input_latent.shape[2], rgb_input_latent.shape[3]

        # Embedding preparation
        emb_masks = null_caption_token.attention_mask
        caption_embs = null_caption_embs
        null_y = null_caption_embs.repeat(1, 1, 1)

        print(f'Finished embedding for image {i + 1}/{batch_size}')

        model_kwargs = {
            'data_info': None,
            'mask': emb_masks,
            'input_latent': rgb_input_latent
        }
        z = torch.randn(1, 4, latent_size_h, latent_size_w, device=device)

        # Initialize DPM-Solver
        dpm_solver = DPMS(
            model.forward_with_dpmsolver,
            condition=caption_embs,
            uncondition=null_y,
            cfg_scale=3.0,
            model_kwargs=model_kwargs
        )

        # Generate samples
        samples = dpm_solver.sample(
            z,
            steps=25,
            order=2,
            skip_type="time_uniform",
            method="multistep",
        )

        samples = samples.to(vae.dtype)
        # Decode the depth from latent space
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

        # Log depth image to wandb
        wandb_images.append(wandb.Image(depth_colored_hwc, caption=f"Depth Image {i}"))

        # Also log real image from rgb_int
        real_image_np = rgb_int[i].detach().cpu().numpy()
        real_image_hwc = chw2hwc(real_image_np)
        wandb_images.append(wandb.Image(real_image_hwc, caption=f"Real Image {i}"))
        del z, rgb_input_latent, samples, depth, depth_pred, depth_colored, real_image_np


    # Log all images to wandb
    wandb.log({f"validation_images_step_{step}": wandb_images, "step": step})

    print("Validation completed and images logged to wandb.")
    gc.collect()
    torch.cuda.empty_cache()


def create_datasets(cfg, rank, world_size):
    loader_seed = 0
    num_workers = 4
    if loader_seed is None:
        loader_generator = None
    else:
        loader_generator = torch.Generator().manual_seed(loader_seed)

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

        # sampler = DistributedMixedBatchSampler(
        #     src_dataset_ls=dataset_ls,
        #     batch_size=cfg.dataloader.effective_batch_size,
        #     drop_last=True,
        #     shuffle=True,
        #     world_size=world_size,
        #     rank=rank,
        #     prob=cfg.dataset.train.prob_ls,
        #     generator=loader_generator,
        # )
        sampler = MixedBatchSampler(
            src_dataset_ls=dataset_ls,
            batch_size=cfg.dataloader.effective_batch_size,
            drop_last=True,
            prob=cfg.dataset.train.prob_ls,
            shuffle=True,
            generator=loader_generator,
        )

        train_loader = DataLoader(
            concat_dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
        )
    else:
        sampler = DistributedSampler(
            train_dataset, 
            num_replicas=world_size, 
            rank=rank, 
            shuffle=True, 
            seed=loader_seed
        )

        train_loader = DataLoader(
            dataset=train_dataset,
            sampler=sampler,
            batch_size=cfg.dataloader.effective_batch_size,
            num_workers=num_workers,
            shuffle=True,
            generator=loader_generator,
        )

    val_dataset = HypersimDataset(
        mode=DatasetMode.TRAIN,  # Since TRAIN changes the shape
        filename_ls_path=cfg.dataset.val.filenames,
        dataset_dir=os.path.join(cfg.paths.base_data_dir, cfg.dataset.val.dir),
        resize_to_hw=cfg.dataset.val.resize_to_hw if 'resize_to_hw' in cfg.dataset.val else None,
        disp_name=cfg.dataset.val.name,
        depth_transform=depth_transform
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.validation.batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    return train_loader, val_loader


def train():
    if config.get('debug_nan', False):
        DebugUnderflowOverflow(model)
        logger.info('NaN debugger registered. Start detecting overflow during training.')

    if accelerator.sync_gradients:
        update_ema(ema, model, decay=0)  # Initialize EMA
        ema.eval()

    time_start, last_tic = time.time(), time.time()
    log_buffer = LogBuffer()
    
    global_step = start_step + 1
    total_iterations = config.num_epochs * len(train_dataloader)  # Calculate total iterations
    epoch = 0

    # Initialize tracking (e.g., WandB)
    if accelerator.is_main_process and args.report_to == "wandb":
        wandb.init(project=args.tracker_project_name, config=config)
        accelerator.init_trackers(args.tracker_project_name, config)

    data_time_start = time.time()
    data_time_all = 0

    # Iteration-based training loop
    for step in tqdm(range(global_step, total_iterations + 1), initial=global_step, total=total_iterations, desc="Training Progress"):
        grad_norm = None
        if step % len(train_dataloader) == 0 or step==1:
            epoch += 1
            # train_dataloader.sampler.set_epoch(epoch)
            train_loader_iter = iter(train_dataloader)
        
        batch = next(train_loader_iter)  # Sample the next batch
        # print(batch["rgb_relative_path"])
        rgb = batch["rgb_norm"].to(device=accelerator.device, dtype=torch.float16)
        depth_gt_for_latent = batch['depth_raw_norm'].to(device=accelerator.device, dtype=torch.float16)

        # Encode inputs
        rgb_input_latent = vae.encode(rgb).latent_dist.sample().mul_(config.scale_factor)
        depth_gt_latent = encode_depth(depth_gt_for_latent, vae)

        del rgb, depth_gt_for_latent
        torch.cuda.empty_cache()

        bs = rgb_input_latent.shape[0]
        y = null_caption_embs.unsqueeze(0).repeat(bs, 1, 1, 1).detach()
        timesteps = torch.randint(0, config.train_sampling_steps, (bs,), device=accelerator.device).long()

        data_time_all += time.time() - data_time_start

        # Training step
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            loss_term = train_diffusion.training_losses(
                model, 
                depth_gt_latent, 
                timesteps, 
                model_kwargs=dict(
                    y=y, 
                    mask=None,
                    input_latent=rgb_input_latent
                )
            )
            loss = loss_term['loss'].mean()
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            update_ema(ema, model.module)
            # lr_scheduler.step()

        # Logging
        # lr = lr_scheduler.get_last_lr()[0]
        logs = {args.loss_report_name: accelerator.gather(loss).mean().item()}
        log_buffer.update(logs)

        if step % config.log_interval == 0:
            avg_time = (time.time() - time_start) / step
            eta = str(datetime.timedelta(seconds=int(avg_time * (total_iterations - step))))
            t_d = data_time_all / config.log_interval

            info = f"Step [{step}/{total_iterations}] ETA: {eta}, " \
                   f"Time Data: {t_d:.3f}, LR: {optimizer.defaults['lr']:.3e}, "
            info += ', '.join([f"{k}:{v:.4f}" for k, v in log_buffer.output.items()])
            logger.info(info)

            last_tic = time.time()
            log_buffer.clear()
            data_time_all = 0
            
        if grad_norm is not None:
            logs.update(grad_norm=accelerator.gather(grad_norm).mean().item())

        # Log to tracking tool (e.g., WandB)
        if accelerator.is_main_process:
            # logs.update(lr=lr)
            accelerator.log(logs, step=step)


        # Save checkpoint periodically
        if config.save_model_steps and step % config.save_model_steps == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                save_checkpoint(
                    os.path.join(config.work_dir, 'checkpoints'),
                    epoch=step // len(train_dataloader),  # Optional: calculate current epoch
                    step=step,
                    model=accelerator.unwrap_model(model),
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler
                )

        # Validation and visualization (optional)
        if config.visualize and step % config.eval_sampling_steps == 0 or step==0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                log_validation(model, val_loader, vae, accelerator.device, step)

        # Final checkpoint after all iterations
        if step == total_iterations:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                save_checkpoint(
                    os.path.join(config.work_dir, 'checkpoints'),
                    epoch=config.num_epochs,
                    step=step,
                    model=accelerator.unwrap_model(model),
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler
                )

        # Prepare for next iteration
        global_step += 1
        data_time_start = time.time()
        del rgb_input_latent, depth_gt_latent, timesteps
        flush()


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("config", type=str, help="config")
    parser.add_argument("--cloud", action='store_true', default=False, help="cloud or local machine")
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the dir to resume the training')
    parser.add_argument('--load-from', default=None, help='the dir to load a ckpt for training')
    parser.add_argument('--local-rank', type=int, default=-1)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--is_depth', action='store_true')
    parser.add_argument(
        "--pipeline_load_from", default='/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/depth_estimation_experiments/PixArt-sigma/output/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers',
        type=str, help="Download for loading text_encoder, "
                       "tokenizer and vae from https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-pixart",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument("--loss_report_name", type=str, default="loss")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = read_config(args.config)
    if args.work_dir is not None:
        config.work_dir = args.work_dir
    if args.resume_from is not None:
        config.load_from = None
        config.resume_from = dict(
            checkpoint=args.resume_from,
            load_ema=False,
            resume_optimizer=True,
            resume_lr_scheduler=True)
    if args.debug:
        config.log_interval = 1
        config.train_batch_size = 2

    os.umask(0o000)
    os.makedirs(config.work_dir, exist_ok=True)

    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400)  # change timeout to avoid a strange NCCL bug
    # Initialize accelerator and tensorboard logging
    if config.use_fsdp:
        init_train = 'FSDP'
        from accelerate import FullyShardedDataParallelPlugin
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
        set_fsdp_env()
        fsdp_plugin = FullyShardedDataParallelPlugin(state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),)
    else:
        init_train = 'DDP'
        fsdp_plugin = None

    even_batches = True
    if config.multi_scale:
        even_batches=False,

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=args.report_to,
        project_dir=os.path.join(config.work_dir, "logs"),
        fsdp_plugin=fsdp_plugin,
        even_batches=even_batches,
        kwargs_handlers=[init_handler]
    )

    log_name = 'train_log.log'
    if accelerator.is_main_process:
        if os.path.exists(os.path.join(config.work_dir, log_name)):
            rename_file_with_creation_time(os.path.join(config.work_dir, log_name))
    logger = get_root_logger(os.path.join(config.work_dir, log_name))

    logger.info(accelerator.state)
    config.seed = init_random_seed(config.get('seed', None))
    set_random_seed(config.seed)

    world_size = get_world_size()
    rank = get_rank()

    logger.info(f"World_size: {get_world_size()}, seed: {config.seed}")
    logger.info(f"Initializing: {init_train} for training")
    image_size = config.image_size  # @param [256, 512]
    latent_size = int(image_size) // 8
    validation_noise = torch.randn(1, 4, latent_size, latent_size, device='cpu') if getattr(config, 'deterministic_validation', False) else None
    pred_sigma = getattr(config, 'pred_sigma', True)
    learn_sigma = getattr(config, 'learn_sigma', True) and pred_sigma
    max_length = config.model_max_length
    kv_compress_config = config.kv_compress_config if config.kv_compress else None
    vae = None
    vae = AutoencoderKL.from_pretrained(config.vae_pretrained, torch_dtype=torch.float16).to(accelerator.device)
    config.scale_factor = vae.config.scaling_factor

    logger.info(f"vae scale factor: {config.scale_factor}")


    model_kwargs = {"pe_interpolation": config.pe_interpolation, "config": config,
                    "model_max_length": max_length, "qk_norm": config.qk_norm,
                    "kv_compress_config": kv_compress_config, "micro_condition": config.micro_condition}
    

    # Check if the .pt files exist, otherwise save them
    save_dir = "output/null_embedding"
    if not (os.path.exists(os.path.join(save_dir, "null_caption_token.pt")) and
            os.path.exists(os.path.join(save_dir, "null_caption_embs.pt"))):
        save_null_caption_embeddings(args.pipeline_load_from, accelerator)

    # Load the saved embeddings and tokens
    null_caption_token, null_caption_embs = load_null_caption_embeddings(save_dir)

    # build models
    train_diffusion = IDDPM(str(config.train_sampling_steps), learn_sigma=learn_sigma, pred_sigma=pred_sigma, snr=config.snr_loss)
    model = build_model(config.model,
                        config.grad_checkpointing,
                        config.get('fp32_attention', False),
                        input_size=latent_size,
                        learn_sigma=learn_sigma,
                        pred_sigma=pred_sigma,
                        **model_kwargs).train()
    logger.info(f"{model.__class__.__name__} Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    if args.load_from is not None:
        config.load_from = args.load_from

    def load_model():
        if config.load_from is not None:
            missing, unexpected = load_checkpoint(
                config.load_from, model, load_ema=config.get('load_ema', False), max_length=max_length)
            logger.warning(f'Missing keys: {missing}')
            logger.warning(f'Unexpected keys: {unexpected}')


    if args.is_depth:
        # For Depth DiT model
        # The state_dict is already modified for depth, so modify the model before loading
        model = _replace_patchembed_proj(model)
        load_model()
    else:
        # For a vanilla DiT model
        # Load the state_dict first, and then modify the model afterwards
        load_model()
        model = _replace_patchembed_proj(model)
    
    ema = deepcopy(model).to(accelerator.device)  # EMA model
    requires_grad(ema, False)

    # model = _replace_patchembed_proj(model)
    
    # prepare for FSDP clip grad norm calculation
    if accelerator.distributed_type == DistributedType.FSDP:
        for m in accelerator._models:
            m.clip_grad_norm_ = types.MethodType(clip_grad_norm_, m)

    # build dataloader
    train_dataloader, val_loader = create_datasets(config.conf_data, rank, world_size)
    logger.info(f"Number of training samples: {len(train_dataloader.dataset)}")
    logger.info(f"Total number of batches: {len(train_dataloader)}")
    logger.info(f"Batch size: {train_dataloader.batch_size}")
    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer["lr"], weight_decay=config.optimizer["weight_decay"])

    # lr_scale_ratio = 1
    # if config.get('auto_lr', None):
    #     lr_scale_ratio = auto_scale_lr(config.train_batch_size * get_world_size() * config.gradient_accumulation_steps,
    #                                    config.optimizer, **config.auto_lr)    
    # lr_scheduler = build_lr_scheduler(config, optimizer, train_dataloader, lr_scale_ratio)

    lr_scheduler = None

    if accelerator.is_main_process:
        tracker_config = dict(vars(config))
        try:
            accelerator.init_trackers(args.tracker_project_name, tracker_config)
        except:
            accelerator.init_trackers(f"tb_{timestamp}")

    start_epoch = 0
    start_step = 0
    skip_step = config.skip_step
    total_steps = len(train_dataloader) * config.num_epochs

    if config.resume_from is not None and config.resume_from['checkpoint'] is not None:
        resume_path = config.resume_from['checkpoint']
        path = os.path.basename(resume_path)
        start_epoch = int(path.replace('.pth', '').split("_")[1]) - 1
        start_step = int(path.replace('.pth', '').split("_")[3])
        _, missing, unexpected = load_checkpoint(**config.resume_from,
                                                 model=model,
                                                 optimizer=optimizer,
                                                 lr_scheduler=lr_scheduler,
                                                 max_length=max_length,
                                                 )

        logger.warning(f'Missing keys: {missing}')
        logger.warning(f'Unexpected keys: {unexpected}')
    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.

    model = accelerator.prepare(model)
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)
    train()

"""
python -m torch.distributed.launch --nproc_per_node=2 --master_port=12345 \
          train_scripts/train_depth.py \
          configs/pixart_sigma_config/PixArt_sigma_xl2_img512_depth.py \
          --load-from /mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/depth_estimation_experiments/PixArt-sigma/output/pretrained_models/PixArt-Sigma-XL-2-1024-MS.pth \
          --work-dir output/depth_512_mixed_training \
          --debug \
          --report_to wandb
"""