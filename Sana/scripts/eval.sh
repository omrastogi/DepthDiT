#!/bin/bash

# Define default values for the parameters
BASE_DATA_DIR="/data/om/data/eval_dataset"
DATASET_CONFIG="configs/dataset/data_nyu_test.yaml"
OUTPUT_DIR="output/batch_eval/nyu_test"
MODEL_PATH="/data/om/Sana/output/debug/checkpoints/epoch_6_step_198000.pth"
CONFIG="configs/sana_config/1024ms/Sana_600M_img1024.yaml"

# Allow overriding parameters via command-line arguments
while [ $# -gt 0 ]; do
  case "$1" in
    --base_data_dir=*)
      BASE_DATA_DIR="${1#*=}"
      ;;
    --dataset_config=*)
      DATASET_CONFIG="${1#*=}"
      ;;
    --output_dir=*)
      OUTPUT_DIR="${1#*=}"
      ;;
    --model_path=*)
      MODEL_PATH="${1#*=}"
      ;;
    --config=*)
      CONFIG="${1#*=}"
      ;;
    *)
      echo "Invalid argument: $1"
      exit 1
  esac
  shift
done

# Echo and run depth inference
echo "Running inference with the following parameters:"
echo "  Config: $CONFIG"
echo "  Base Data Directory: $BASE_DATA_DIR"
echo "  Model Path: $MODEL_PATH"
echo "  Output Directory: $OUTPUT_DIR/prediction"
echo "  Data Config: $DATASET_CONFIG"

python scripts/inference_depth.py \
    --config "$CONFIG" \
    --base_data_dir "$BASE_DATA_DIR" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR/prediction" \
    --data_config "$DATASET_CONFIG"

# Echo and run evaluation
echo "Running evaluation with the following parameters:"
echo "  Base Data Directory: $BASE_DATA_DIR"
echo "  Dataset Config: $DATASET_CONFIG"
echo "  Alignment: least_square"
echo "  Prediction Directory: $OUTPUT_DIR/prediction"
echo "  Output Directory: $OUTPUT_DIR/eval_metric"

python scripts/eval.py \
    --base_data_dir "$BASE_DATA_DIR" \
    --dataset_config "$DATASET_CONFIG" \
    --alignment least_square \
    --prediction_dir "$OUTPUT_DIR/prediction" \
    --output_dir "$OUTPUT_DIR/eval_metric"
