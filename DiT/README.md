## Training and Inference for Depth Estimation

We provide scripts for training and inference of depth estimation using DiT models.

### Inference

```bash
python inference_depth.py \
  --model DiT-XL/2 \
  --image-size 512 \
  --batch-size 9 \
  --num-sampling-steps 25 \
  --ckpt /mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/depth_estimation_experiments/DiT/results/model_vkitti_hypersim_4_epoch_multires/checkpoints/0014000.pt \
  --image-path /mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/depth_estimation_experiments/DiT/data/lab_img \
  --output-path /mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/depth_estimation_experiments/DiT/results/lab_mixed_trained_4_channel_cfg
```

### Training

To train a new DiT model for depth estimation, use the `train_depth.py` script. This script supports distributed training and various configurations. For example, to train a DiT-XL/2 model on 2 GPU, run:

```bash
torchrun --nnodes=1 --nproc_per_node=2  train_depth.py \
--model DiT-XL/2 \
--valid-mask-loss \
--epochs 6 \
--validation-every 1000 \
--global-batch-size 20 \
--ckpt-every 2000 \
--image-size 512 \
--data-path /mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/DiT/data/imagenet/train
```


### Evaluation

- Benchmark Inference

```bash
python infer.py \
--model DiT-XL/2 \
--image-size 512 \
--batch-size 10 \
--num-sampling-steps 50 \
--ensemble-size 10 \
--dataset-config config/dataset/data_nyu_test.yaml \
--base-data-dir /mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/depth_estimation_experiments/Marigold/eval_dataset \
--ckpt /mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/depth_estimation_experiments/DiT/results/model_vkitti_hypersim_4_epoch_multires/checkpoints/0014000.pt \
--output-dir /mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/depth_estimation_experiments/DiT/results/model_vkitti_hypersim_4_epoch_multires/batch_eval/nyu_test/prediction 
```

- Evaluation

```bash
python eval.py \
--base_data_dir /mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/depth_estimation_experiments/Marigold/eval_dataset \
--dataset_config config/dataset/data_nyu_test.yaml \
--alignment least_square \
--prediction_dir /mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/depth_estimation_experiments/DiT/results/model_vkitti_hypersim_4_epoch_multires/batch_eval/nyu_test/prediction \
--output_dir /mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/depth_estimation_experiments/DiT/results/model_vkitti_hypersim_4_epoch_multires/batch_eval/nyu_test/eval_metric \
```