# Attention Visualization Guide

This guide explains how to visualize transformer attention weights from the AoANet model on individual images.

## Overview

The attention visualization tool allows you to:
- Capture and visualize attention weights from any layer in the model
- See which parts of an image the model focuses on
- Compare attention patterns across different layers and heads
- Export visualizations as overlay images and heatmaps

## Quick Start

### Prerequisites

Ensure you have the required dependencies:
```bash
pip install torch torchvision numpy opencv-python Pillow
```

### Basic Usage

Visualize attention on a single image:

```bash
python tools/visualize_attention.py \
    --image path/to/your/image.jpg \
    --infos_path path/to/infos_aoanet.pkl \
    --model path/to/model.pth
```

This will:
1. Load your trained model
2. Process the image through the model
3. Capture attention weights from all attention layers
4. Generate heatmap visualizations
5. Save results to `attn_vis/` directory

### Output Files

The tool generates three types of files for each attention layer:

1. **Attention Overlay**: `attn_map_layer_{i}_{layer_name}.png`
   - Original image with attention heatmap overlaid
   - Shows which image regions receive most attention
   
2. **Pure Heatmap**: `heatmap_layer_{i}_{layer_name}.png`
   - Just the attention heatmap without the original image
   - Useful for analyzing attention patterns independently
   
3. **Raw Data** (optional): `attn_raw_layer_{i}.npy`
   - Raw attention tensor saved as NumPy array
   - Enabled with `--save-raw` flag

## Advanced Usage

### Visualize Specific Layer

Focus on a particular attention layer:

```bash
python tools/visualize_attention.py \
    --image path/to/image.jpg \
    --infos_path path/to/infos.pkl \
    --model path/to/model.pth \
    --layer 0
```

### Visualize Specific Attention Head

View attention from a single head (without averaging):

```bash
python tools/visualize_attention.py \
    --image path/to/image.jpg \
    --infos_path path/to/infos.pkl \
    --model path/to/model.pth \
    --head 2 \
    --no-avg-heads
```

### Process Multiple Images

Use a bash loop to process multiple images:

```bash
for img in data/images/*.jpg; do
    basename=$(basename "$img" .jpg)
    python tools/visualize_attention.py \
        --image "$img" \
        --infos_path log/infos.pkl \
        --model log/model.pth \
        --output "attn_vis_${basename}"
done
```

### Save Raw Attention Tensors

For further analysis or custom visualization:

```bash
python tools/visualize_attention.py \
    --image path/to/image.jpg \
    --infos_path path/to/infos.pkl \
    --model path/to/model.pth \
    --save-raw
```

Then load and analyze in Python:

```python
import numpy as np

# Load raw attention tensor
attn = np.load('attn_vis/attn_raw_layer_0.npy')

# Analyze attention patterns
print(f"Attention shape: {attn.shape}")
print(f"Mean attention: {attn.mean():.4f}")
print(f"Max attention: {attn.max():.4f}")
```

## Understanding the Visualizations

### Color Coding

The heatmaps use a JET colormap by default:
- **Red/Yellow**: High attention (model focuses here)
- **Green**: Medium attention
- **Blue**: Low attention (model ignores these regions)

### Interpretation

- **Early Layers**: Often show broad, distributed attention patterns
- **Middle Layers**: May focus on specific object boundaries or features
- **Late Layers**: Typically concentrate on the most relevant image regions

### CLS Token Attention

The visualization shows the **CLS token's attention to image patches**:
- CLS token aggregates information from the entire image
- Its attention pattern reveals which image regions contribute most to the final representation
- This is the most commonly visualized attention pattern in vision transformers

## Python API Usage

For integration into custom scripts:

```python
from tools.visualize_attention import visualize_attention_on_image
import models
import misc.utils as utils

# Load model
with open('log/infos.pkl', 'rb') as f:
    infos = utils.pickle_load(f)

opt = infos['opt']
opt.vocab = infos['vocab']
model = models.setup(opt)
model.load_state_dict(torch.load('log/model.pth'))

# Visualize attention
visualize_attention_on_image(
    model=model,
    image_path='data/sample.jpg',
    device='cuda',
    output_dir='my_visualizations',
    layer_idx=None,      # All layers
    head_idx=None,       # Average all heads
    avg_heads=True,
    save_raw=False
)
```

## Examples

See the `tools/` directory for example scripts:
- `tools/example_usage.sh` - Bash script examples
- `tools/example_usage.py` - Python code examples

Run examples:
```bash
# Show bash examples
bash tools/example_usage.sh

# Show Python examples
python tools/example_usage.py
```

## Troubleshooting

### Issue: "No attention maps captured"

**Cause**: The model's attention modules aren't being detected.

**Solution**: 
1. Check that your model uses `MultiHeadedDotAttention` or `MultiHeadedAttention`
2. Verify attention weights are stored in the `attn` attribute
3. Modify the hook registration if needed

### Issue: "Model forward pass fails"

**Cause**: The model requires additional inputs beyond the image tensor.

**Solution**:
- For caption models, the script tries to call `model.encode()`
- You may need to modify the forward pass code for your specific model
- Check what inputs your model's forward method expects

### Issue: "Attention visualization looks wrong"

**Possible causes**:
1. Model doesn't use CLS token at position 0
2. Number of patches doesn't match expected grid
3. Wrong layer or head selected

**Solutions**:
- Try visualizing different layers with `--layer`
- Try specific heads with `--head`
- Check model architecture and token ordering

### Issue: "Out of memory"

**Solution**: Use CPU instead of GPU:
```bash
python tools/visualize_attention.py \
    --image path/to/image.jpg \
    --infos_path path/to/infos.pkl \
    --model path/to/model.pth \
    --device cpu
```

## Technical Details

### Attention Hook Mechanism

The tool uses PyTorch's `register_forward_hook` to capture attention weights:
1. Hooks are registered on attention modules before forward pass
2. During forward pass, hooks capture attention tensors
3. Hooks are removed after capturing

### Supported Attention Modules

Out of the box, the tool supports:
- `MultiHeadedDotAttention` (AoANet)
- `MultiHeadedAttention` (Transformer)
- `torch.nn.MultiheadAttention` (PyTorch standard)

### Attention Processing Pipeline

1. **Capture**: Raw attention tensor `[batch, heads, seq, seq]`
2. **Head Aggregation**: Average across heads or select specific head
3. **CLS Extraction**: Extract CLS token attention to patches
4. **Reshape**: Convert flat patches to 2D grid
5. **Resize**: Interpolate to original image resolution
6. **Normalize**: Scale to [0, 1] range
7. **Colormap**: Apply color mapping (JET by default)
8. **Overlay**: Blend with original image

## Contributing

To add support for custom attention modules:

1. Modify the `register_attention_hooks` function
2. Add your module type to the `module_types` tuple
3. Ensure attention is stored in a standard attribute (e.g., `module.attn`)

Example:
```python
from mymodel import CustomAttention

def register_attention_hooks(model, module_types=None):
    if module_types is None:
        module_types = (
            MultiHeadedDotAttention,
            MultiHeadedAttention,
            CustomAttention  # Add your type here
        )
    # ... rest of the function
```

## Further Reading

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer paper
- [Attention on Attention for Image Captioning](https://arxiv.org/abs/1908.06954) - AoANet paper
- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) - Vision Transformer paper

## Citation

If you use this visualization tool in your research, please cite the AoANet paper:

```bibtex
@inproceedings{huang2019attention,
  title={Attention on Attention for Image Captioning},
  author={Huang, Lun and Wang, Wenmin and Chen, Jie and Wei, Xiao-Yong},
  booktitle={International Conference on Computer Vision},
  year={2019}
}
```
