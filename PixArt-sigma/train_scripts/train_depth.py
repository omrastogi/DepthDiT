import argparse
import datetime
import gc
import os
import sys
import time
import types
import warnings
from pathlib import Path
import wandb
from omegaconf import OmegaConf
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
from src.dataset.hypersim_dataset import HypersimDataset
from src.utils.image_utils import decode_depth, colorize_depth_maps, chw2hwc
warnings.filterwarnings("ignore")  # ignore warning


def set_fsdp_env():
    os.environ["ACCELERATE_USE_FSDP"] = 'true'
    os.environ["FSDP_AUTO_WRAP_POLICY"] = 'TRANSFORMER_BASED_WRAP'
    os.environ["FSDP_BACKWARD_PREFETCH"] = 'BACKWARD_PRE'
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = 'PixArtBlock'

# Modified from Marigold: https://github.com/prs-eth/Marigold/blob/62413d56099d36573b2de1eb8c429839734b7782/src/trainer/marigold_trainer.py#L387
def encode_depth(depth_in, vae):
    # stack depth into 3-channel
    stacked = stack_depth_images(depth_in)
    # encode using VAE encoder
    depth_latent = vae.encode(stacked).latent_dist.sample().mul_(vae.config.scaling_factor)
    return depth_latent

def stack_depth_images(depth_in):
    if 4 == len(depth_in.shape):
        stacked = depth_in.repeat(1, 3, 1, 1)
    elif 3 == len(depth_in.shape):
        stacked = depth_in.unsqueeze(1)
        stacked = depth_in.repeat(1, 3, 1, 1)
    return stacked

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
            cfg_scale=4.5,
            model_kwargs=model_kwargs
        )

        # Generate samples
        samples = dpm_solver.sample(
            z,
            steps=14,
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

    # cfg = config_dataset

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
            batch_size=cfg.dataloader.effective_batch_size,
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
        mode=DatasetMode.EVAL,  # Since TRAIN changes the shape
        filename_ls_path=cfg.dataset.val.filenames,
        dataset_dir=os.path.join(cfg.paths.base_data_dir, cfg.dataset.val.dir),
        # resize_to_hw=cfg.dataset.val.resize_to_hw,
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
        logger.info('NaN debugger registered. Start to detect overflow during training.')
    time_start, last_tic = time.time(), time.time()
    log_buffer = LogBuffer()

    global_step = start_step + 1
    if args.report_to == "wandb":
        wandb.init(project=args.tracker_project_name, config=config)
        accelerator.init_trackers(args.tracker_project_name, config)
        
    # Now you train the model
    for epoch in range(start_epoch + 1, config.num_epochs + 1):
        data_time_start= time.time()
        data_time_all = 0
        for step, batch in enumerate(train_dataloader):
            # print(batch["rgb_relative_path"])
            # print(torch.cuda.memory_summary())
            if step < skip_step:
                global_step += 1
                continue    # skip data in the resumed ckpt

            # if self.args.valid_mask_loss: #TODO add valid mask later p -2
            #     valid_mask_for_latent = batch['valid_mask_raw'].to(self.device)
            #     invalid_mask = ~valid_mask_for_latent
            #     valid_mask_down = ~torch.max_pool2d(invalid_mask.float(), 8, 8).bool().repeat((1, 4, 1, 1))
            # else:
            #     valid_mask_down = None

            rgb = batch["rgb_norm"].to(device=accelerator.device, dtype=torch.float16)
            depth_gt_for_latent = batch['depth_raw_norm'].to(device=accelerator.device, dtype=torch.float16)

            #TODO experiment with vae.encode(rgb).latent_dist.mode() p
            #TODO Also need to see if there need to be a different scaling factor
            rgb_input_latent = vae.encode(rgb).latent_dist.sample().mul_(config.scale_factor)
            depth_gt_latent = encode_depth(depth_gt_for_latent, vae)

            del rgb, depth_gt_for_latent
            torch.cuda.empty_cache()

            """
            data_info = {'img_hw': tensor([[1024., 1024.], [1024., 1024.]], device='cuda:0'), 'aspect_ratio': tensor([1., 1.], device='cuda:0', dtype=torch.float64), 'mask_type': ['null', 'null']
            """

            bs = rgb_input_latent.shape[0]

            y = null_caption_embs.unsqueeze(0).repeat(bs, 1, 1, 1).detach()
            # Sample a random timestep for each image
            # bs = rgb_input_latent.shape[0]
            timesteps = torch.randint(0, config.train_sampling_steps, (bs,), device=accelerator.device).long()
            grad_norm = None
            data_time_all += time.time() - data_time_start
            with accelerator.accumulate(model):
                optimizer.zero_grad()  # Make sure gradients are cleared only once
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
                accelerator.backward(loss)  # Should only be called once per step
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
                lr_scheduler.step()

            lr = lr_scheduler.get_last_lr()[0]
            logs = {args.loss_report_name: accelerator.gather(loss).mean().item()}
            if grad_norm is not None:
                logs.update(grad_norm=accelerator.gather(grad_norm).mean().item())
            log_buffer.update(logs)
            if (step + 1) % config.log_interval == 0 or (step + 1) == 1:
                t = (time.time() - last_tic) / config.log_interval
                t_d = data_time_all / config.log_interval
                avg_time = (time.time() - time_start) / (global_step + 1)
                eta = str(datetime.timedelta(seconds=int(avg_time * (total_steps - global_step - 1))))
                eta_epoch = str(datetime.timedelta(seconds=int(avg_time * (len(train_dataloader) - step - 1))))
                log_buffer.average()

                info = f"Step/Epoch [{global_step}/{epoch}][{step + 1}/{len(train_dataloader)}]:total_eta: {eta}, " \
                    f"epoch_eta:{eta_epoch}, time_all:{t:.3f}, time_data:{t_d:.3f}, lr:{lr:.3e}, "
                info += f's:({model.module.h}, {model.module.w}), ' if hasattr(model, 'module') else f's:({model.h}, {model.w}), '

                info += ', '.join([f"{k}:{v:.4f}" for k, v in log_buffer.output.items()])
                logger.info(info)
                last_tic = time.time()
                log_buffer.clear()
                data_time_all = 0
            logs.update(lr=lr)
            accelerator.log(logs, step=global_step)

            global_step += 1
            data_time_start = time.time()
            del rgb_input_latent, depth_gt_latent, timesteps
            flush()
            if config.save_model_steps and global_step % config.save_model_steps == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    os.umask(0o000)
                    save_checkpoint(os.path.join(config.work_dir, 'checkpoints'),
                                    epoch=epoch,
                                    step=global_step,
                                    model=accelerator.unwrap_model(model),
                                    optimizer=optimizer,
                                    lr_scheduler=lr_scheduler
                                    )
            if config.visualize and (global_step % config.eval_sampling_steps == 0 or (step + 1) == 1):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    log_validation(model, val_loader, vae, accelerator.device, global_step)

        if epoch % config.save_model_epochs == 0 or epoch == config.num_epochs:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                os.umask(0o000)
                save_checkpoint(os.path.join(config.work_dir, 'checkpoints'),
                                epoch=epoch,
                                step=global_step,
                                model=accelerator.unwrap_model(model),
                                optimizer=optimizer,
                                lr_scheduler=lr_scheduler
                                )
        accelerator.wait_for_everyone()


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
    parser.add_argument(
        "--pipeline_load_from", default='output/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers',
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

    # if accelerator.is_main_process:
    #     config.dump(os.path.join(config.work_dir, 'config.py'))

    # if accelerator.is_main_process:
    #     with open(os.path.join(config.work_dir, 'config.py'), 'w') as f:
    #         f.write(OmegaConf.to_yaml(config))

    # logger.info(f"Config: \n{config.pretty_text}")
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
    

    max_sequence_length = 300
    tokenizer = T5Tokenizer.from_pretrained(args.pipeline_load_from, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(
        args.pipeline_load_from, subfolder="text_encoder", torch_dtype=torch.float16).to(accelerator.device)

    null_caption_token = tokenizer("", max_length=max_sequence_length, padding="max_length", truncation=True, return_tensors="pt").to(accelerator.device)
    null_caption_embs = text_encoder(null_caption_token.input_ids, attention_mask=null_caption_token.attention_mask)[0]
    del tokenizer, text_encoder
    flush()


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
    if config.load_from is not None:
        missing, unexpected = load_checkpoint(
            config.load_from, model, load_ema=config.get('load_ema', False), max_length=max_length)
        logger.warning(f'Missing keys: {missing}')
        logger.warning(f'Unexpected keys: {unexpected}')
    
    model = _replace_patchembed_proj(model)
    #TODO: Add the replacement code here- 1

    # prepare for FSDP clip grad norm calculation
    if accelerator.distributed_type == DistributedType.FSDP:
        for m in accelerator._models:
            m.clip_grad_norm_ = types.MethodType(clip_grad_norm_, m)

    # build dataloader
    train_dataloader, val_loader = create_datasets(config.conf_data, rank, world_size)


    lr_scale_ratio = 1
    if config.get('auto_lr', None):
        lr_scale_ratio = auto_scale_lr(config.train_batch_size * get_world_size() * config.gradient_accumulation_steps,
                                       config.optimizer, **config.auto_lr)
    optimizer = build_optimizer(model, config.optimizer)
    lr_scheduler = build_lr_scheduler(config, optimizer, train_dataloader, lr_scale_ratio)

    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

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
python -m torch.distributed.launch --nproc_per_node=1 --master_port=12345 \
          train_scripts/train_depth.py \
          configs/pixart_sigma_config/PixArt_sigma_xl2_img512_depth.py \
          --load-from output/pretrained_models/PixArt-Sigma-XL-2-512-MS.pth \
          --work-dir output/your_first_pixart-exp \
          --debug \
          --report_to wandb
"""