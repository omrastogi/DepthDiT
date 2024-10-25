
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

```bash
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=12345 \
          train_scripts/train_depth.py \
          configs/pixart_sigma_config/PixArt_sigma_xl2_img512_depth.py \
          --load-from output/pretrained_models/PixArt-Sigma-XL-2-512-MS.pth \
          --work-dir output/your_first_pixart-exp \
          --debug \
          --report_to wandb
```
---