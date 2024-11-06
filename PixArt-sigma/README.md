
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
pip install scipy=1.13.1, tabulate=0.9.0
```

### 1.2 Download pretrained checkpoint
```bash
# SDXL-VAE, T5 checkpoints
git lfs install
git clone https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers output/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers

# PixArt-Sigma checkpoints
python tools/download.py # environment eg. HF_ENDPOINT=https://hf-mirror.com can use for HuggingFace mirror
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
# Evaluation 

## Batch Inference 
```bash
# NYUv2 dataset
python scripts/batch_inference_depth.py \
    --model_path /data/om/models/depth_512_mixed_training/checkpoints/epoch_11_step_52000.pth \
    --base_data_dir /data/om/data/eval_dataset \
    --config_path configs/dataset/data_nyu_test.yaml \
    --output_dir /data/om/models/depth_512_mixed_training/batch_eval/epoch_11_step_52000/prediction \
    --sampling_algo dpm-solver \
    --ensemble_size 10 \
    --batch_size 10 \
    --step 25 \
    --image_size 512 \
    --is_depth \
```
```bash
python eval.py \
--base_data_dir /data/om/data/eval_dataset \
--dataset_config configs/dataset/data_nyu_test.yaml \
--alignment least_square \
--prediction_dir /data/om/models/depth_512_mixed_training/batch_eval/epoch_11_step_52000 \
--output_dir /data/om/models/depth_512_mixed_training/batch_eval/epoch_11_step_52000/metric \
```
