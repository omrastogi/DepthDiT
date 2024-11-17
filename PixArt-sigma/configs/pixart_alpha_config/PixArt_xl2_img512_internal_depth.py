from omegaconf import OmegaConf

_base_ = ['../PixArt_xl2_internal.py']
#TODO create a data_config.yaml
conf_data = OmegaConf.load(
    "/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/depth_estimation_experiments/DiT/PixArt-sigma/configs/depth_data_config/training_config.yaml"
)
valid_mask_loss = True
image_size = 512

# model setting
model = 'PixArt_XL_2'
fp32_attention = True
load_from = None
vae_pretrained = "stabilityai/sd-vae-ft-ema"
pe_interpolation = 1.0

# training setting
use_fsdp=False   # if use FSDP mode
num_workers=10
train_batch_size = 38 # 32
num_epochs = 200 # 3
gradient_accumulation_steps = 1
grad_checkpointing = True
gradient_clip = 0.01
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-4)
lr_schedule = 'constant'
lr_schedule_args = dict(num_warmup_steps=500)

eval_sampling_steps = 200
log_interval = 20
save_model_epochs=1
work_dir = 'output/debug'

visualize = True

# multi res noise
multi_res_noise = True
multi_res_noise_strength = True
multi_res_noise_annealing = 0.9
multi_res_noise_downscale_strategy = "original" 

