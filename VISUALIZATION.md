# Attention Visualization for AoANet

This guide explains how to visualize attention weights in the AoANet model during inference.

## Overview

The attention visualization feature allows you to see which parts of an image the model focuses on when generating each word in the caption. This helps in understanding the model's behavior and interpretability.

## Features

- **Attention Weight Extraction**: Captures attention weights during caption generation
- **Heatmap Generation**: Creates color-coded heatmaps showing attention distribution
- **Image Overlay**: Overlays attention heatmaps on the original image
- **Word-level Visualization**: Generates separate visualizations for each word in the caption
- **Summary View**: Creates a grid view showing attention for multiple words at once

## Usage

### Basic Usage

To visualize attention for images in a folder:

```bash
python visualize.py \
    --model path/to/model.pth \
    --infos_path path/to/infos.pkl \
    --image_folder path/to/images \
    --output_dir vis/attention \
    --num_images 5
```

### Parameters

#### Required Parameters

- `--model`: Path to the trained model checkpoint (.pth file)
- `--infos_path`: Path to the model info file (.pkl file)
- `--image_folder`: Folder containing images to visualize

#### Optional Parameters

- `--output_dir`: Directory to save visualizations (default: `vis/attention`)
- `--num_images`: Number of images to process, -1 for all (default: 1)
- `--beam_size`: Beam size for caption generation (default: 1, greedy)
- `--sample_method`: Sampling method - 'greedy' or 'sample' (default: greedy)
- `--temperature`: Temperature for sampling (default: 1.0)
- `--cnn_model`: CNN model for features (default: resnet101)
- `--coco_json`: Optional COCO JSON file for image metadata

### Example

```bash
# Visualize attention for a single image
python visualize.py \
    --model log/log_aoanet_rl/model.pth \
    --infos_path log/log_aoanet_rl/infos_aoanet.pkl \
    --image_folder data/images \
    --num_images 1

# Visualize with beam search
python visualize.py \
    --model log/log_aoanet_rl/model.pth \
    --infos_path log/log_aoanet_rl/infos_aoanet.pkl \
    --image_folder data/images \
    --beam_size 5 \
    --num_images 3
```

## Output

For each processed image, the script generates:

1. **Individual Word Visualizations**: One image per word showing:
   - Original image (left)
   - Attention heatmap overlay (right)
   - Word label as title

   Format: `{image_name}_word_{index}_{word}.png`

2. **Summary Visualization**: A grid showing attention for all words

   Format: `{image_name}_summary.png`

## Understanding the Visualizations

- **Red/Yellow regions**: High attention - the model is focusing on these areas
- **Blue/Purple regions**: Low attention - the model is not focusing much here
- **Color intensity**: Indicates the strength of attention

### Example Interpretation

For a caption "A cat sitting on a couch":
- Word "cat": High attention (red) on the cat region
- Word "couch": High attention (red) on the couch region
- Word "sitting": Attention distributed between cat and couch

## Implementation Details

### Modified Files

1. **models/AttModel.py**: 
   - Added `self.attn` attribute to the `Attention` class
   - Stores attention weights computed during forward pass
   - Minimal change that doesn't affect existing functionality

### New Files

1. **vis_utils.py**: Visualization utilities
   - `AttentionHook`: Captures attention during forward pass
   - `capture_attention_weights()`: Generates caption and captures attention
   - `get_attention_weights_from_sequence()`: Extracts attention for pre-generated captions
   - `resize_attention_to_image()`: Resizes attention map to image dimensions
   - `create_attention_heatmap()`: Creates heatmap overlay
   - `visualize_attention_for_sequence()`: Main visualization function

2. **visualize.py**: Main visualization script
   - Loads model and processes images
   - Generates captions and captures attention
   - Creates and saves visualizations

3. **test_vis_utils.py**: Unit tests for visualization utilities
   - Tests all major components
   - Can be run with: `python test_vis_utils.py`

## Technical Notes

### Attention Mechanism

The AoANet model uses an Attention-on-Attention (AoA) mechanism with multi-headed attention. The visualization:

1. Captures attention weights from the decoder's attention module
2. For multi-headed attention, averages across heads
3. Maps attention to spatial locations on the image
4. Generates heatmap visualizations

### Attention Size

The attention size (number of regions) depends on the visual features:
- Grid features (e.g., ResNet): Typically 7×7 = 49 regions
- Bottom-up features: Variable number of object regions

The visualization automatically handles different attention sizes.

### Limitations

- **Beam Search**: During beam search with beam_size > 1, attention capture may not work perfectly due to the complex beam expansion process. In such cases, the script falls back to extracting attention by re-running the model with the generated sequence.
- **Multi-stage Attention**: The visualization focuses on the decoder's attention to image features. Other attention mechanisms (e.g., self-attention in the refiner) are not currently visualized.

## Testing

Run the test suite to verify the visualization utilities:

```bash
python test_vis_utils.py
```

All tests should pass, confirming that:
- Attention maps are correctly resized
- Heatmaps are properly generated
- Visualizations are saved successfully
- Attention hooks work correctly

## Dependencies

The visualization feature requires:
- PyTorch
- NumPy
- OpenCV (opencv-python)
- Matplotlib
- Pillow (PIL)

Install with:
```bash
pip install torch numpy opencv-python matplotlib pillow
```

## Troubleshooting

### No attention weights captured

If the script reports "Could not extract attention weights":
- Ensure the model has an attention module (check model architecture)
- Try using greedy decoding (beam_size=1) instead of beam search
- The script will automatically try to re-extract attention from the sequence

### Image not found

If images are not found:
- Check that `--image_folder` points to the correct directory
- Ensure image paths in metadata match actual file locations
- The script will try to find images by filename if full path fails

### Memory issues

For large numbers of images or high-resolution visualizations:
- Process images in smaller batches (use `--num_images`)
- Reduce batch size (currently fixed at 1 for visualization)
- Consider reducing visualization resolution

## Future Enhancements

Potential improvements for the visualization feature:
- Interactive visualization with web interface
- Video generation showing attention evolution
- Comparison mode for different models
- Attention to specific image regions (e.g., detected objects)
- Export attention data in JSON format for external analysis
