# Attention Visualization Tools

This directory contains tools for visualizing attention weights from the AoANet model.

## visualize_attention.py

A script to visualize transformer attention weights on a single image during evaluation.

### Features

- Registers forward hooks on attention modules to capture attention weights
- Supports both `MultiHeadedDotAttention` (AoANet) and `MultiHeadedAttention` (Transformer)
- Processes attention tensors: averages heads or selects specific heads
- Extracts CLS token attention to image patches
- Reshapes to patch grid and resizes to image resolution
- Generates color heatmaps and overlays on original images
- Saves visualizations to disk

### Requirements

The script requires the following Python packages:
- torch
- torchvision
- numpy
- opencv-python (cv2)
- Pillow (PIL)

### Usage

#### Basic Usage

Visualize attention on a single image:

```bash
python tools/visualize_attention.py \
    --image path/to/image.jpg \
    --infos_path log/infos_aoanet.pkl \
    --model log/model.pth
```

#### Advanced Options

Specify output directory and device:

```bash
python tools/visualize_attention.py \
    --image data/sample.jpg \
    --infos_path log/infos_aoanet.pkl \
    --model log/model.pth \
    --output my_visualizations \
    --device cuda
```

Visualize specific layer and attention head:

```bash
python tools/visualize_attention.py \
    --image data/sample.jpg \
    --infos_path log/infos_aoanet.pkl \
    --model log/model.pth \
    --layer 0 \
    --head 2 \
    --no-avg-heads
```

Save raw attention tensors:

```bash
python tools/visualize_attention.py \
    --image data/sample.jpg \
    --infos_path log/infos_aoanet.pkl \
    --model log/model.pth \
    --save-raw
```

### Command-Line Arguments

- `--image`: Path to input image (required)
- `--model`: Path to model checkpoint .pth file (required)
- `--infos_path`: Path to infos pickle file (required)
- `--output`: Output directory for visualizations (default: `attn_vis`)
- `--device`: Device to use: `cuda` or `cpu` (default: `cuda`)
- `--layer`: Specific layer index to visualize (default: all layers)
- `--head`: Specific attention head to visualize (default: average all heads)
- `--no-avg-heads`: Do not average attention heads (requires `--head`)
- `--save-raw`: Save raw attention tensors as .npy files

### Output

The script generates the following files in the output directory:

1. **Overlay images**: `attn_map_layer_{i}_{layer_name}.png`
   - Original image with attention heatmap overlaid
   
2. **Heatmap images**: `heatmap_layer_{i}_{layer_name}.png`
   - Pure attention heatmap with colormap applied
   
3. **Raw attention tensors** (if `--save-raw`): `attn_raw_layer_{i}.npy`
   - Raw attention weights as numpy arrays

### How It Works

1. **Hook Registration**: The script registers forward hooks on attention modules in the model. It automatically detects:
   - `MultiHeadedDotAttention` (used in AoANet)
   - `MultiHeadedAttention` (used in Transformer)
   - Standard `torch.nn.MultiheadAttention`

2. **Forward Pass**: Runs the model on the input image to capture attention weights.

3. **Attention Processing**:
   - Extracts CLS token's attention to image patches
   - Averages across attention heads (or selects specific head)
   - Reshapes to 2D grid matching image patch layout
   - Resizes to original image dimensions

4. **Visualization**:
   - Normalizes attention values to [0, 1]
   - Applies JET colormap
   - Overlays on original image with transparency

### Troubleshooting

#### No attention maps captured

If you see "No attention maps captured", it means the hooks didn't capture any attention weights. This can happen if:

1. The model doesn't use the expected attention module types
2. The attention weights are not stored in the `attn` attribute during forward pass
3. The model requires additional inputs beyond the image tensor

**Solution**: Modify the `register_attention_hooks` function to match your model's attention mechanism.

#### Model forward pass fails

If the forward pass fails, it might be because:

1. The model expects additional inputs (e.g., captions for decoder)
2. The model architecture requires specific input formats

**Solution**: The script tries to call `model.encode()` for caption models, but you may need to modify the forward pass logic for your specific model.

#### Wrong attention visualization

If the visualization looks incorrect:

1. Check if your model uses CLS token at index 0
2. Verify the number of patches matches expected grid size
3. Try visualizing different layers or heads

### Example Output

For a typical image, the script will produce:

```
Using device: cuda
Loading infos from: log/infos_aoanet.pkl
Setting up model: aoa
Loading model weights from: log/model.pth
Registered hook on: core.refiner.layers.0.self_attn
Registered hook on: core.refiner.layers.1.self_attn
Registered hook on: core.refiner.layers.2.self_attn
...
Loaded image: data/sample.jpg, size: (640, 480)
Running forward pass...
Captured 12 attention maps

Processing attention map 0 from layer: core.refiner.layers.0.self_attn
  Shape: torch.Size([1, 8, 36, 36])
  Saved overlay: attn_vis/attn_map_layer_0_core_refiner_layers_0_self_attn.png
  Saved heatmap: attn_vis/heatmap_layer_0_core_refiner_layers_0_self_attn.png
...

Visualization complete! Results saved to: attn_vis
```

### Notes

- The script assumes the model uses a CLS token at position 0 followed by image patches
- Patches are assumed to be arranged in a square or near-square grid
- If the number of patches is not a perfect square, padding is added
- The default colormap is JET, which shows hot regions in red and cool regions in blue
