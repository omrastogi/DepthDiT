## Training and Inference for Depth Estimation

We provide scripts for training and inference of depth estimation using DiT models.
```bash
python depth_dit.py --model DiT-XL/2 --image-size 512 --ckpt /mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/DiT/results/002-DiT-XL-2/checkpoints/0000150.pt
```

### Training

To train a new DiT model for depth estimation, use the `train_depth.py` script. This script supports distributed training and various configurations. For example, to train a DiT-XL/2 model with 1 GPU, run:

```bash
torchrun --nnodes=1 --nproc_per_node=1 train_depth.py --model DiT-XL/2 --epochs 800 --validation-every 200 --global-batch-size 4 --ckpt-every 1600 --image-size 512 --data-path /mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/DiT/data/imagenet/train
```