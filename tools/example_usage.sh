#!/bin/bash
# Example usage of the attention visualization tool
# This script demonstrates various ways to use visualize_attention.py

# Note: Update these paths to match your setup
MODEL_PATH="log/log_aoanet_rl/model.pth"
INFOS_PATH="log/log_aoanet_rl/infos_aoanet.pkl"
IMAGE_PATH="data/sample_image.jpg"

echo "================================================"
echo "Attention Visualization Tool - Example Usage"
echo "================================================"
echo ""

# Example 1: Basic usage - visualize all layers
echo "Example 1: Basic usage (all layers)"
echo "-----------------------------------"
echo "Command:"
echo "python tools/visualize_attention.py \\"
echo "    --image $IMAGE_PATH \\"
echo "    --infos_path $INFOS_PATH \\"
echo "    --model $MODEL_PATH \\"
echo "    --output attn_vis_basic"
echo ""
# Uncomment to run:
# python tools/visualize_attention.py \
#     --image "$IMAGE_PATH" \
#     --infos_path "$INFOS_PATH" \
#     --model "$MODEL_PATH" \
#     --output attn_vis_basic

# Example 2: Visualize specific layer
echo "Example 2: Visualize specific layer (layer 0)"
echo "----------------------------------------------"
echo "Command:"
echo "python tools/visualize_attention.py \\"
echo "    --image $IMAGE_PATH \\"
echo "    --infos_path $INFOS_PATH \\"
echo "    --model $MODEL_PATH \\"
echo "    --output attn_vis_layer0 \\"
echo "    --layer 0"
echo ""
# Uncomment to run:
# python tools/visualize_attention.py \
#     --image "$IMAGE_PATH" \
#     --infos_path "$INFOS_PATH" \
#     --model "$MODEL_PATH" \
#     --output attn_vis_layer0 \
#     --layer 0

# Example 3: Visualize specific attention head
echo "Example 3: Visualize specific attention head (head 0)"
echo "------------------------------------------------------"
echo "Command:"
echo "python tools/visualize_attention.py \\"
echo "    --image $IMAGE_PATH \\"
echo "    --infos_path $INFOS_PATH \\"
echo "    --model $MODEL_PATH \\"
echo "    --output attn_vis_head0 \\"
echo "    --head 0 \\"
echo "    --no-avg-heads"
echo ""
# Uncomment to run:
# python tools/visualize_attention.py \
#     --image "$IMAGE_PATH" \
#     --infos_path "$INFOS_PATH" \
#     --model "$MODEL_PATH" \
#     --output attn_vis_head0 \
#     --head 0 \
#     --no-avg-heads

# Example 4: Save raw attention tensors
echo "Example 4: Save raw attention tensors"
echo "--------------------------------------"
echo "Command:"
echo "python tools/visualize_attention.py \\"
echo "    --image $IMAGE_PATH \\"
echo "    --infos_path $INFOS_PATH \\"
echo "    --model $MODEL_PATH \\"
echo "    --output attn_vis_raw \\"
echo "    --save-raw"
echo ""
# Uncomment to run:
# python tools/visualize_attention.py \
#     --image "$IMAGE_PATH" \
#     --infos_path "$INFOS_PATH" \
#     --model "$MODEL_PATH" \
#     --output attn_vis_raw \
#     --save-raw

# Example 5: Use CPU instead of CUDA
echo "Example 5: Use CPU device"
echo "-------------------------"
echo "Command:"
echo "python tools/visualize_attention.py \\"
echo "    --image $IMAGE_PATH \\"
echo "    --infos_path $INFOS_PATH \\"
echo "    --model $MODEL_PATH \\"
echo "    --output attn_vis_cpu \\"
echo "    --device cpu"
echo ""
# Uncomment to run:
# python tools/visualize_attention.py \
#     --image "$IMAGE_PATH" \
#     --infos_path "$INFOS_PATH" \
#     --model "$MODEL_PATH" \
#     --output attn_vis_cpu \
#     --device cpu

echo ""
echo "================================================"
echo "To run these examples:"
echo "1. Update the paths at the top of this script"
echo "2. Uncomment the example you want to run"
echo "3. Execute: bash tools/example_usage.sh"
echo "================================================"
