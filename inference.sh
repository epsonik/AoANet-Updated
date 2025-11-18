#!/bin/bash
# Example script for running inference with evaluation
# Modify paths as needed for your setup

# Model paths
MODEL_PATH="${MODEL_PATH:-log/log_aoanet_rl/model.pth}"
INFOS_PATH="${INFOS_PATH:-log/log_aoanet_rl/infos_aoanet.pkl}"

# Data paths
IMAGE_FOLDER="${IMAGE_FOLDER:-/path/to/val2014}"
REFERENCE_CAPTIONS="${REFERENCE_CAPTIONS:-captions_val2014.json}"

# Model configuration
CNN_MODEL="${CNN_MODEL:-densenet161}"
BEAM_SIZE="${BEAM_SIZE:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"

# Output
OUTPUT_JSON="${OUTPUT_JSON:-predictions.json}"

# Run inference
python inference.py \
    --model "$MODEL_PATH" \
    --infos_path "$INFOS_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --reference_captions "$REFERENCE_CAPTIONS" \
    --cnn_model "$CNN_MODEL" \
    --beam_size "$BEAM_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --output_json "$OUTPUT_JSON" \
    --verbose 1
