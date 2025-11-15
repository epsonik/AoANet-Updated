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
- `--coco_annotations`: Path to COCO annotations for metric calculation (default: `coco-caption/annotations/captions_val2014.json`)

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

For each processed image, the script creates a subdirectory within the output folder named after the image file (without extension). All visualizations for that image are saved in this subdirectory.

**Directory Structure**: `{output_dir}/{image_name}/`

For example, if the input image is `example_image.jpg`, all outputs will be saved in `{output_dir}/example_image/`.

Within each image subdirectory, the script generates:

1. **Original Image**: A copy of the input image saved for reference
   
   Format: `original.jpg`

2. **Individual Word Visualizations**: One image per word showing attention overlay on the image
   - Attention heatmap overlaid directly on the image
   - Uses jet colormap with alpha blending
   - No titles or borders

   Format: `{index}_{word}.png` (e.g., `0_a.png`, `1_cat.png`, `2_sitting.png`)

3. **Summary Visualization**: A horizontal grid showing the original image and attention for all words

   Format: `{image_name}_summary.png`

4. **Evaluation Metrics** (if COCO annotations are provided): CSV file containing caption quality metrics

   Format: `evaluation_metrics.csv` (in the output directory)

### Example Output Structure

When processing two images `cat.jpg` and `dog.jpg` with output directory `vis/attention`:

```
vis/attention/
├── cat/
│   ├── original.jpg
│   ├── 0_a.png
│   ├── 1_cat.png
│   ├── 2_sitting.png
│   ├── 3_on.png
│   ├── 4_couch.png
│   └── cat_summary.png
└── dog/
    ├── original.jpg
    ├── 0_a.png
    ├── 1_dog.png
    ├── 2_playing.png
    └── dog_summary.png
```

This organization makes it easy to find all visualizations related to a specific input image.

## Understanding the Visualizations

- **Red/Yellow regions**: High attention - the model is focusing on these areas
- **Blue/Purple regions**: Low attention - the model is not focusing much here
- **Color intensity**: Indicates the strength of attention

### Example Interpretation

For a caption "A cat sitting on a couch":
- Word "cat": High attention (red) on the cat region
- Word "couch": High attention (red) on the couch region
- Word "sitting": Attention distributed between cat and couch

## Evaluation Metrics

When processing images from the COCO dataset, the script can automatically calculate standard image captioning metrics for each generated caption. This feature helps evaluate the quality of the model's predictions.

### Metrics Calculated

For each image, the following metrics are computed by comparing the predicted caption against ground-truth reference captions:

- **BLEU-1, BLEU-2, BLEU-3, BLEU-4**: N-gram overlap metrics (higher is better, range 0-1)
- **METEOR**: Metric considering synonyms and stemming (higher is better, range 0-1)
- **CIDEr**: Consensus-based metric weighted by TF-IDF (higher is better, typically 0-10)
- **ROUGE_L**: Longest common subsequence based metric (higher is better, range 0-1)

### Using Metric Calculation

To enable metric calculation, ensure the COCO annotations file is available:

```bash
python visualize.py \
    --model path/to/model.pth \
    --infos_path path/to/infos.pkl \
    --image_folder path/to/coco/images \
    --coco_annotations coco-caption/annotations/captions_val2014.json \
    --output_dir vis/attention \
    --num_images 5
```

The script will:
1. Extract the COCO image ID from the filename (e.g., `COCO_val2014_000000391895.jpg` → 391895)
2. Load reference captions for that image from the annotations
3. Calculate metrics comparing predicted vs. reference captions
4. Print metrics to console for each image
5. Save all metrics to `evaluation_metrics.csv` in the output directory

### CSV Output Format

The `evaluation_metrics.csv` file contains:
- `image_id`: COCO image ID
- `predicted_caption`: Generated caption by the model
- `BLEU_1`, `BLEU_2`, `BLEU_3`, `BLEU_4`: BLEU scores
- `METEOR`: METEOR score
- `CIDEr`: CIDEr score
- `ROUGE_L`: ROUGE_L score

Example CSV content:
```csv
image_id,predicted_caption,BLEU_1,BLEU_2,BLEU_3,BLEU_4,METEOR,CIDEr,ROUGE_L
391895,a bicycle with a clock as the front wheel,0.8571,0.7143,0.5714,0.4286,0.3456,1.2345,0.6789
203564,a dog sitting on a couch,0.9000,0.7500,0.6000,0.4500,0.4123,1.5678,0.7234
```

### Requirements for Metric Calculation

The metric calculation feature requires:
- `pycocoevalcap` library (included in the coco-caption submodule)
- `pycocotools` (included in the coco-caption submodule)
- COCO annotations file (captions_val2014.json)
- Images must be from the COCO dataset with standard naming convention

If any of these requirements are not met, the script will skip metric calculation and only generate visualizations.

### Example with Metrics

```bash
# Process COCO validation images with metric calculation
python visualize.py \
    --model log/log_aoanet_rl/model.pth \
    --infos_path log/log_aoanet_rl/infos_aoanet.pkl \
    --image_folder /path/to/coco/val2014 \
    --coco_annotations coco-caption/annotations/captions_val2014.json \
    --output_dir vis/attention_with_metrics \
    --num_images 10
```

Console output will include:
```
Processing image 1: COCO_val2014_000000391895.jpg
Generated caption: a bicycle with a clock as the front wheel
Calculating metrics for image ID: 391895
  BLEU-1: 0.8571, BLEU-2: 0.7143, BLEU-3: 0.5714, BLEU-4: 0.4286
  METEOR: 0.3456, CIDEr: 1.2345, ROUGE_L: 0.6789
  Metrics saved to: vis/attention_with_metrics/evaluation_metrics.csv
```

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
   - `resize_attention_to_image()`: Resizes attention map using smooth pyramid upscaling
   - `create_attention_heatmap()`: Creates heatmap overlay with jet colormap
   - `visualize_attention_for_sequence()`: Main visualization function (overlay style)

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

## Demo

Try the visualization feature without a trained model using synthetic attention patterns:

```bash
python demo_visualization.py
```

This creates example visualizations showing different attention patterns (top-left, center, right, bottom, distributed) for the words in a sample caption. The demo helps you understand what the visualizations look like before running on a real model.

## Testing

### Visualization Tests

Run the test suite to verify the visualization utilities:

```bash
python test_vis_utils.py
```

All tests should pass, confirming that:
- Attention maps are correctly resized
- Heatmaps are properly generated
- Visualizations are saved successfully
- Attention hooks work correctly

### Metric Calculation Tests

Run the test suite to verify the metric calculation functionality:

```bash
python test_metrics.py
```

All tests should pass, confirming that:
- Image ID extraction from filenames works correctly
- CSV output is formatted properly
- Metric calculation functions correctly (if dependencies are available)

## Dependencies

The visualization feature requires:
- PyTorch
- NumPy
- OpenCV (opencv-python)
- Matplotlib
- Pillow (PIL)
- scikit-image (for smooth upscaling)

Install with:
```bash
pip install torch numpy opencv-python matplotlib pillow scikit-image
```

For metric calculation, additional dependencies are required:
- pycocoevalcap (included in coco-caption submodule)
- pycocotools (included in coco-caption submodule)
- Java (for METEOR metric calculation)

To set up the coco-caption submodule:
```bash
git submodule update --init --recursive
```

Or manually clone:
```bash
git clone https://github.com/ruotianluo/coco-caption.git
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
