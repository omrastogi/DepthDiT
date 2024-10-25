#!/usr/bin/env bash
set -x

# Use specified checkpoint path, otherwise, default value
CKPT=${1:-"checkpoints/vkitti_hypersim_mixed_training_continued/checkpoints/0010000.pt"}

# Use specified subfolder, otherwise, default value
subfolder=${2:-"nyu_test"}

# Use specified output directory base, otherwise, default value
OUTPUT_DIR_BASE=${3:-"checkpoints/vkitti_hypersim_mixed_training_continued/10000_batch_eval"}

# Use specified batch size, otherwise, default value
BATCH_SIZE=${4:-10}

# Use specified number of sampling steps, otherwise, default value
NUM_SAMPLING_STEPS=${5:-50}

# Use specified ensemble size, otherwise, default value
ENSEMBLE_SIZE=${6:-10}

# Use specified base data directory, otherwise, default value
BASE_DATA_DIR=${7:-"/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/depth_estimation_experiments/Marigold/eval_dataset"}

MODEL="DiT-XL/2"
IMAGE_SIZE=512


echo "CKPT: $CKPT"
echo "BASE_DATA_DIR: $BASE_DATA_DIR"
echo "OUTPUT_DIR_BASE: $OUTPUT_DIR_BASE"
echo "Running inference with the following command:"
echo "python infer.py --model $MODEL --image-size $IMAGE_SIZE --batch-size $BATCH_SIZE --num-sampling-steps $NUM_SAMPLING_STEPS --ensemble-size $ENSEMBLE_SIZE --dataset-config config/dataset/data_nyu_test.yaml --base-data-dir $BASE_DATA_DIR --ckpt $CKPT --output-dir $OUTPUT_DIR_BASE/$subfolder/prediction"

# Infer
python infer.py \
  --fp16 \
  --model $MODEL \
  --image-size $IMAGE_SIZE \
  --batch-size $BATCH_SIZE \
  --num-sampling-steps $NUM_SAMPLING_STEPS \
  --ensemble-size $ENSEMBLE_SIZE \
  --dataset-config config/dataset/data_nyu_test.yaml \
  --base-data-dir $BASE_DATA_DIR \
  --ckpt $CKPT \
  --output-dir $OUTPUT_DIR_BASE/$subfolder/prediction

# Eval
python eval.py \
  --base_data_dir $BASE_DATA_DIR \
  --dataset_config config/dataset/data_nyu_test.yaml \
  --alignment least_square \
  --prediction_dir $OUTPUT_DIR_BASE/$subfolder/prediction \
  --output_dir $OUTPUT_DIR_BASE/$subfolder/eval_metric
