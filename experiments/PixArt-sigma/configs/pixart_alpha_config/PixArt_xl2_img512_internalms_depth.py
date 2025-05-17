from omegaconf import OmegaConf


_base_ = ['../PixArt_xl2_internal.py']
conf_data = OmegaConf.load(
    "/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/depth_estimation_experiments/DiT/PixArt-sigma/configs/depth_data_config/training_config.yaml"
)
valid_mask_loss = True
image_size = 512

# model setting
model = 'PixArtMS_XL_2'     # model for multi-scale training
fp32_attention = True
load_from = None
vae_pretrained = "stabilityai/sd-vae-ft-ema"
aspect_ratio_type = 'ASPECT_RATIO_512'         # base aspect ratio [ASPECT_RATIO_512 or ASPECT_RATIO_256]
multi_scale = True     # if use multiscale dataset model training
pe_interpolation = 1.0

# dataloader settings
num_workers = 2
train_batch_size = 8   # max 40 for PixArt-xL/2 when grad_checkpoint
val_batch_size = 3

# training setting
num_epochs = 200 # 3
num_iterations = 40000
gradient_accumulation_steps = 8
grad_checkpointing = True

# optimizer setting
gradient_clip = 1.0
lr = 1e-4
weight_decay=1e-4
optimizer = dict(type='AdamW', lr=lr, weight_decay=weight_decay)

lr_scheduler = True 
cosine_annealing = False
start_step = 0

# Validation and Logs
eval_sampling_steps = 500
visualize = True
log_interval = 4
save_model_steps = 2000
work_dir = '/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/depth_estimation_experiments/DiT/PixArt-sigma/output'

# multi res noise
multi_res_noise = True
multi_res_noise_strength = True
multi_res_noise_annealing = 0.9
multi_res_noise_downscale_strategy = "original" 

