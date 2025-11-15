# Image Captioning Inference with Evaluation

## Overview

The `inference.py` script provides a streamlined way to generate captions for images and evaluate them against reference captions from the COCO dataset. This script is particularly useful for:

- Generating captions for COCO validation set images
- Evaluating generated captions using standard metrics (BLEU, METEOR, ROUGE, CIDEr)
- Processing custom image directories with automatic feature extraction

## Features

- **Automatic Feature Extraction**: Uses CNN models (DenseNet, ResNet, etc.) to extract features on-the-fly
- **Image ID Extraction**: Automatically extracts COCO image IDs from filenames (e.g., `COCO_val2014_000000123456.jpg`)
- **Standard Evaluation Metrics**: Calculates BLEU-1/2/3/4, METEOR, ROUGE_L, CIDEr, and SPICE scores
- **Flexible Input**: Works with COCO-formatted JSON or raw image directories
- **Batch Processing**: Efficiently processes multiple images in batches

## Requirements

Before running the script, ensure you have:

1. A trained model checkpoint (`.pth` file)
2. Model info file (`.pkl` file)
3. Reference captions file (`captions_val2014.json`)
4. COCO validation images in a directory
5. COCO evaluation tools (coco-caption submodule initialized)

## Usage

### Basic Usage

```bash
python inference.py \
    --model log/log_aoanet_rl/model.pth \
    --infos_path log/log_aoanet_rl/infos_aoanet.pkl \
    --image_folder /path/to/val2014 \
    --reference_captions captions_val2014.json
```

### Advanced Options

```bash
python inference.py \
    --model log/log_aoanet_rl/model.pth \
    --infos_path log/log_aoanet_rl/infos_aoanet.pkl \
    --image_folder /path/to/val2014 \
    --reference_captions captions_val2014.json \
    --cnn_model resnet101 \
    --beam_size 5 \
    --batch_size 10 \
    --output_json results/predictions.json \
    --verbose 1
```

## Command-Line Arguments

### Required Arguments

- `--model`: Path to the trained model checkpoint (`.pth` file)
- `--infos_path`: Path to the model info file (`.pkl` file)
- `--image_folder`: Path to directory containing input images

### Optional Arguments

- `--reference_captions`: Path to reference captions file (default: `captions_val2014.json`)
  - Set to empty string to skip evaluation
- `--cnn_model`: CNN model for feature extraction (default: `densenet161`)
  - Options: `densenet161`, `densenet121`, `densenet169`, `densenet201`, `resnet101`, `resnet152`, `regnet`, `inception`
- `--coco_json`: Path to COCO-style JSON file with image metadata (optional)
- `--beam_size`: Beam size for caption generation (default: `2`)
- `--batch_size`: Number of images to process in each batch (default: `1`)
- `--output_json`: Path to save predictions as JSON (optional)
- `--verbose`: Print captions as they are generated (default: `1`)

## Input Image Format

The script expects images with COCO-style filenames:

```
COCO_val2014_000000391895.jpg
COCO_val2014_000000123456.jpg
```

The image ID is automatically extracted from the filename to match with reference captions.

## Output

### Console Output

The script displays:

1. Progress information during caption generation
2. Generated captions (if `--verbose 1`)
3. Evaluation metrics in a formatted table

Example output:

```
============================================================
EVALUATION RESULTS
============================================================

Metrics:
  BLEU-1: 0.8055
  BLEU-2: 0.6523
  BLEU-3: 0.5097
  BLEU-4: 0.3914
  METEOR: 0.2901
  ROUGE_L: 0.5890
  CIDEr: 1.2892
  SPICE: 0.2268

============================================================
```

### JSON Output

If `--output_json` is specified, predictions are saved in COCO results format:

```json
[
  {
    "image_id": 391895,
    "caption": "A woman is holding a tennis racket on a court.",
    "file_path": "/path/to/COCO_val2014_000000391895.jpg"
  },
  ...
]
```

## Examples

### Example 1: Quick Evaluation

Evaluate model on a subset of validation images:

```bash
python inference.py \
    --model pretrained_model.pth \
    --infos_path pretrained_info.pkl \
    --image_folder val2014_subset \
    --reference_captions captions_val2014.json
```

### Example 2: Generate Captions Without Evaluation

Generate captions for custom images without evaluation:

```bash
python inference.py \
    --model pretrained_model.pth \
    --infos_path pretrained_info.pkl \
    --image_folder my_custom_images \
    --reference_captions "" \
    --output_json my_captions.json
```

### Example 3: High-Quality Caption Generation

Use larger beam size for better quality captions:

```bash
python inference.py \
    --model pretrained_model.pth \
    --infos_path pretrained_info.pkl \
    --image_folder val2014 \
    --reference_captions captions_val2014.json \
    --beam_size 5 \
    --batch_size 10
```

## Evaluation Metrics

The script calculates the following standard image captioning metrics:

- **BLEU-1/2/3/4**: Measures n-gram overlap with reference captions
- **METEOR**: Considers synonyms and word stems
- **ROUGE_L**: Longest common subsequence-based metric
- **CIDEr**: Consensus-based metric weighted by TF-IDF
- **SPICE**: Evaluates semantic content using scene graphs

## Differences from eval.py

The `inference.py` script differs from `eval.py` in several ways:

| Feature | inference.py | eval.py |
|---------|--------------|---------|
| Feature Extraction | On-the-fly from raw images | Pre-extracted features required |
| Input Format | Raw image files | HDF5 feature files |
| Use Case | Quick inference on new images | Benchmarking with preprocessed data |
| Setup | Simpler, no feature preprocessing | Requires feature extraction setup |

## Troubleshooting

### "No predictions match reference captions"

This occurs when image IDs cannot be extracted from filenames. Ensure your images follow the COCO naming convention:
- `COCO_val2014_000000XXXXXX.jpg`

### CUDA Out of Memory

Reduce batch size:
```bash
--batch_size 1
```

### coco-caption Import Error

Initialize git submodules:
```bash
git submodule update --init --recursive
```

## Performance Tips

1. **Batch Size**: Increase `--batch_size` for faster processing on GPU
2. **Beam Size**: Use smaller beam size (2-3) for speed, larger (5+) for quality
3. **CNN Model**: DenseNet161 provides good trade-off between speed and accuracy
4. **GPU Memory**: Monitor GPU memory usage and adjust batch size accordingly

## Citation

If you use this code in your research, please cite:

```
@inproceedings{huang2019attention,
  title={Attention on Attention for Image Captioning},
  author={Huang, Lun and Wang, Wenmin and Chen, Jie and Wei, Xiao-Yong},
  booktitle={International Conference on Computer Vision},
  year={2019}
}
```
