from omegaconf import OmegaConf


_base_ = ['../PixArt_xl2_internal.py']

#TODO create a data_config.yaml
conf_data = OmegaConf.load(
    "/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/depth_estimation_experiments/DiT/PixArt-sigma/configs/depth_data_config/overfitting_config.yaml"
)

image_size = 512

# model setting
model = 'PixArtMS_XL_2'
mixed_precision = 'fp16'  # ['fp16', 'fp32', 'bf16']
fp32_attention = False
load_from = "/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/depth_estimation_experiments/PixArt-sigma/output/pretrained_models/PixArt-Sigma-XL-2-512-MS.pth"  # https://huggingface.co/PixArt-alpha/PixArt-Sigma
resume_from = None
vae_pretrained = "/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/depth_estimation_experiments/PixArt-sigma/output/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers/vae"  # sdxl vae
aspect_ratio_type = 'ASPECT_RATIO_512'
multi_scale = True  # if use multiscale dataset model training
pe_interpolation = 1.0

# training setting
num_workers = 10
train_batch_size = 3  # 48 as default
num_epochs = 1000  # 3
gradient_accumulation_steps = 4
grad_checkpointing = True
gradient_clip = 0.01
optimizer = dict(type='CAMEWrapper', lr=2e-5, weight_decay=0.0, betas=(0.9, 0.999, 0.9999), eps=(1e-30, 1e-16))
lr_schedule_args = dict(num_warmup_steps=1000)

eval_sampling_steps = 500
visualize = True
log_interval = 20
save_model_epochs = 3000
save_model_steps = 10000
work_dir = '/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/depth_estimation_experiments/PixArt-sigma/output/debug'

# pixart-sigma
scale_factor = 0.13025
real_prompt_ratio = 0.5
model_max_length = 300
class_dropout_prob = 0.1
