from omegaconf import OmegaConf


_base_ = ['../PixArt_xl2_internal.py']

#TODO create a data_config.yaml
conf_data = OmegaConf.load(
    "configs/depth_data_config/training_config.yaml"
)
valid_mask_loss = True

image_size = 512

# model setting
model = 'PixArtMS_XL_2'
mixed_precision = 'no'  # ['no', 'fp16', 'fp32', 'bf16']
fp32_attention = False
load_from = "output/pretrained_models/PixArt-Sigma-XL-2-512-MS.pth"  # https://huggingface.co/PixArt-alpha/PixArt-Sigma
resume_from = None
vae_pretrained = "output/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers/vae"  # sdxl vae
aspect_ratio_type = 'ASPECT_RATIO_512'
multi_scale = True  # if use multiscale dataset model training
pe_interpolation = 1.0

# dataloader settings
num_workers=2
train_batch_size = 4   # max 40 for PixArt-xL/2 when grad_checkpoint
val_batch_size = 3

# training setting
num_epochs = 200  # 3
num_iterations = 60000
gradient_accumulation_steps = 4
grad_checkpointing = True
gradient_clip = 1.0

# lr
lr = 2e-4
weight_decay=0.1
optimizer = dict(type='AdamW', lr=lr, weight_decay=weight_decay)

lr_scheduler = True
cosine_annealing = False
start_step=500
warmup_steps=500
stop_step=40000
final_lr = 1e-5

eval_sampling_steps = 500
visualize = True
log_interval = 5
save_model_steps = 2000
work_dir = 'output'

# multi res noise
multi_res_noise = False
multi_res_noise_strength = True
multi_res_noise_annealing = 0.9
multi_res_noise_downscale_strategy = "original"
 
# pixart-sigma
scale_factor = 0.13025
real_prompt_ratio = 0.5
model_max_length = 300
class_dropout_prob = 0.2
