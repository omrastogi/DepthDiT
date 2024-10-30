
# 🔧 Dependencies and Installation

- Python >= 3.9 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 2.0.1+cu11.7](https://pytorch.org/)

```bash
conda create -n pixart python==3.9.0
conda activate pixart
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

git clone https://github.com/PixArt-alpha/PixArt-sigma.git
cd PixArt-sigma
pip install -r requirements.txt
```

---

# Training

### Training from scratch

```bash

python -m torch.distributed.launch --nproc_per_node=2 --master_port=12345 \
          train_scripts/train_depth.py \
          configs/pixart_sigma_config/PixArt_sigma_xl2_img512_depth.py \
          --load-from /mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/depth_estimation_experiments/PixArt-sigma/output/pretrained_models/PixArt-Sigma-XL-2-1024-MS.pth \
          --work-dir output/depth_512_mixed_training \
          --debug \
          --report_to wandb
```

### Training a depth model
```bash
python -m torch.distributed.launch --nproc_per_node=2 --master_port=12345 \
          train_scripts/train_depth.py \
          configs/pixart_sigma_config/PixArt_sigma_xl2_img512_depth.py \
          --load-from /mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/depth_estimation_experiments/DiT/PixArt-sigma/output/depth_mixed_training/checkpoints/epoch_3_step_14000.pth \
          --work-dir output/depth_512_mixed_training \
          --is_depth \
          --debug \  
          --report_to_wandb
```
---

# Inference 
```bash
python scripts/inference_depth.py \
    --txt_file asset/samples_new.txt \
    --model_path /mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/depth_estimation_experiments/DiT/PixArt-sigma/output/depth_512_mixed_training/checkpoints/epoch_5_step_20000.pth \
    --cfg_scale 3.0 \
    --is_depth \
    --image_size 512

```