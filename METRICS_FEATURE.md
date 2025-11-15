# Image Captioning Metrics Feature

This document describes the new metric calculation feature added to `visualize.py`.

## Overview

The `visualize.py` script now automatically calculates standard image captioning metrics (BLEU, METEOR, CIDEr, ROUGE_L) when processing images from the COCO dataset. This allows for immediate evaluation of model performance while generating visualizations.

## Metrics Calculated

For each image, the following metrics are computed by comparing the predicted caption against ground-truth reference captions from the COCO dataset:

- **BLEU-1, BLEU-2, BLEU-3, BLEU-4**: N-gram overlap metrics (0-1, higher is better)
- **METEOR**: Metric considering synonyms and stemming (0-1, higher is better)
- **CIDEr**: Consensus-based metric weighted by TF-IDF (typically 0-10, higher is better)
- **ROUGE_L**: Longest common subsequence based metric (0-1, higher is better)

## Implementation

### New Functions in visualize.py

1. **`calculate_metrics_for_image(image_id, predicted_caption, coco_annotations)`**
   - Calculates all metrics for a single image
   - Uses the pycocoevalcap library (same as eval_utils.py)
   - Returns a dictionary of metric scores

2. **`extract_image_id_from_filename(filename)`**
   - Extracts COCO image ID from filename
   - Handles standard COCO format: `COCO_val2014_000000391895.jpg` → `391895`
   - Also handles simple numeric filenames: `391895.jpg` → `391895`

3. **`save_metrics_to_csv(csv_file, image_id, predicted_caption, metrics)`**
   - Appends metrics to CSV file
   - Creates header on first write
   - UTF-8 encoding for international characters in captions

### Integration

The metric calculation is integrated into the main image processing loop:

1. After generating a caption for each image
2. Extract the numeric COCO image ID from the filename or existing ID
3. If valid, calculate metrics using reference captions
4. Print metrics to console
5. Append metrics to CSV file

### Command-Line Interface

New argument:
- `--coco_annotations`: Path to COCO annotations file (default: `coco-caption/annotations/captions_val2014.json`)

If the annotations file doesn't exist or an image is not found in the annotations, metric calculation is skipped for that image (visualization still proceeds).

## Usage

### Basic Usage with Metrics

```bash
python visualize.py \
    --model log/log_aoanet_rl/model.pth \
    --infos_path log/log_aoanet_rl/infos_aoanet.pkl \
    --image_folder /path/to/coco/val2014 \
    --coco_annotations coco-caption/annotations/captions_val2014.json \
    --output_dir vis/attention_with_metrics \
    --num_images 10
```

### Output

1. **Console Output** (per image):
   ```
   Processing image 1: COCO_val2014_000000391895.jpg
   Generated caption: a bicycle with a clock as the front wheel
   Calculating metrics for image ID: 391895
     BLEU-1: 0.8571, BLEU-2: 0.7143, BLEU-3: 0.5714, BLEU-4: 0.4286
     METEOR: 0.3456, CIDEr: 1.2345, ROUGE_L: 0.6789
     Metrics saved to: vis/attention_with_metrics/evaluation_metrics.csv
   ```

2. **CSV File** (`evaluation_metrics.csv`):
   ```csv
   image_id,predicted_caption,BLEU_1,BLEU_2,BLEU_3,BLEU_4,METEOR,CIDEr,ROUGE_L
   391895,a bicycle with a clock as the front wheel,0.8571,0.7143,0.5714,0.4286,0.3456,1.2345,0.6789
   203564,a dog sitting on a couch,0.9000,0.7500,0.6000,0.4500,0.4123,1.5678,0.7234
   ```

3. **Attention Visualizations** (existing functionality, unchanged)

## Dependencies

The metric calculation requires:
- `pycocoevalcap` (from coco-caption submodule)
- `pycocotools` (from coco-caption submodule)
- Java runtime (for METEOR metric)
- matplotlib, numpy (for pycocotools)

To set up:
```bash
git submodule update --init --recursive
# or
git clone https://github.com/ruotianluo/coco-caption.git
```

If dependencies are not available, the script will skip metric calculation and only generate visualizations.

## Testing

### Test Suite

Run `test_metrics.py` to verify functionality:

```bash
python test_metrics.py
```

Tests include:
- Image ID extraction from various filename formats
- CSV writing and appending
- Metric calculation (if dependencies available)

### Demo

Run `demo_metrics.py` to see example output:

```bash
python demo_metrics.py
```

This creates a sample CSV and demonstrates the feature without requiring a trained model or dependencies.

## Design Decisions

### Minimal Changes

The implementation makes minimal modifications to existing code:
- All new code is in new functions
- Existing visualization functionality is unchanged
- Metric calculation is optional (skipped if dependencies unavailable)

### Per-Image Calculation

Metrics are calculated per-image rather than in batch:
- Allows immediate feedback as images are processed
- Consistent with the visualization workflow (one image at a time)
- CSV is appended incrementally (safe for interruptions)

### CSV Output Format

CSV format was chosen for:
- Easy import into spreadsheet applications
- Human-readable
- Machine-readable for analysis scripts
- UTF-8 encoding for international captions

### Error Handling

The implementation gracefully handles:
- Missing dependencies (skips metric calculation)
- Missing COCO annotations file
- Images not in COCO dataset
- Invalid image IDs
- File I/O errors

## Comparison with eval_utils.py

The `eval_utils.py` module has a `language_eval()` function for batch evaluation. The new feature in `visualize.py` is different:

| Feature | eval_utils.py | visualize.py (new) |
|---------|---------------|-------------------|
| Scope | Batch evaluation of all images | Per-image evaluation during visualization |
| Output | Overall metrics + JSON | Per-image metrics + CSV |
| Use case | Model evaluation | Interactive analysis + visualization |
| Integration | Evaluation pipeline | Visualization pipeline |

Both use the same underlying `pycocoevalcap` library for consistency.

## Future Enhancements

Potential improvements:
- Add confidence intervals for metrics
- Support custom annotation files
- Add more metrics (SPICE, etc.)
- Aggregate statistics in CSV (mean, std)
- Interactive web interface for exploring results
- Export metrics in JSON format

## References

- BLEU: Papineni et al., "BLEU: a Method for Automatic Evaluation of Machine Translation"
- METEOR: Banerjee and Lavie, "METEOR: An Automatic Metric for MT Evaluation"
- CIDEr: Vedantam et al., "CIDEr: Consensus-based Image Description Evaluation"
- ROUGE: Lin, "ROUGE: A Package for Automatic Evaluation of Summaries"

## License

This feature follows the same license as the AoANet-Updated repository.
