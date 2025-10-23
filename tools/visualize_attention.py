#!/usr/bin/env python
"""
Visualize transformer attention weights on a single image during evaluation.

This script:
- Registers forward hooks on the model's attention modules to capture attention weights
- Runs a single forward pass on a chosen image
- Postprocesses attention tensors: average heads (or choose a head), select the CLS token's 
  attention to patches, reshape to patch grid, resize to image resolution, normalize
- Converts to a color heatmap and overlays on the original image
- Saves one or more visualizations to disk

Usage:
    python tools/visualize_attention.py --image path/to/img.jpg --infos_path path/to/infos.pkl --model path/to/model.pth
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import math
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T

# Add parent directory to path to import models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import models
import misc.utils as utils


def register_attention_hooks(model, module_types=None, layer_idx=None):
    """
    Register forward hooks on attention modules to capture attention weights.
    
    Args:
        model: The model to register hooks on
        module_types: Tuple of module types to hook (e.g., (nn.MultiheadAttention,))
        layer_idx: Optional specific layer index to hook (None = all layers)
    
    Returns:
        hooks: List of hook handles
        attn_maps: List to store captured attention maps
        layer_names: List of layer names that were hooked
    """
    if module_types is None:
        # Default to common attention module types in this codebase
        from models.AoAModel import MultiHeadedDotAttention
        from models.TransformerModel import MultiHeadedAttention
        module_types = (MultiHeadedDotAttention, MultiHeadedAttention, nn.MultiheadAttention)
    
    attn_maps = []
    hooks = []
    layer_names = []
    
    def hook(module, input, output):
        """Hook function to capture attention weights."""
        attn = None
        
        # Try to get attention from module's attn attribute (common pattern)
        if hasattr(module, 'attn') and module.attn is not None:
            attn = module.attn
        # For MultiheadAttention, check if output is a tuple with attention weights
        elif isinstance(output, tuple) and len(output) > 1:
            attn = output[1]
        # For some modules, attention might be stored as an attribute after forward
        elif hasattr(module, '_attn'):
            attn = module._attn
            
        if attn is not None:
            attn_maps.append(attn.detach().cpu())
    
    # Register hooks on matching modules
    for name, mod in model.named_modules():
        if isinstance(mod, module_types):
            if layer_idx is None or str(layer_idx) in name:
                hooks.append(mod.register_forward_hook(hook))
                layer_names.append(name)
                print(f"Registered hook on: {name}")
    
    return hooks, attn_maps, layer_names


def load_image(image_path, device, size=None):
    """
    Load and preprocess an image for model input.
    
    Args:
        image_path: Path to image file
        device: Device to load image on
        size: Optional resize dimension (int or tuple)
    
    Returns:
        pil_img: Original PIL Image
        tensor: Preprocessed tensor ready for model
        orig_size: Original image size (width, height)
    """
    img = Image.open(image_path).convert("RGB")
    orig_size = img.size  # (w, h)
    
    # Standard ImageNet normalization
    transforms = [
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    
    if size is not None:
        transforms.insert(0, T.Resize(size))
    
    preprocess = T.Compose(transforms)
    tensor = preprocess(img).unsqueeze(0).to(device)
    
    return img, tensor, orig_size


def cls_attention_to_map(attn_tensor, orig_size, head_idx=None, avg_heads=True):
    """
    Convert CLS token attention to 2D heatmap.
    
    Args:
        attn_tensor: Attention tensor from model (various shapes possible)
        orig_size: Original image size (width, height)
        head_idx: Optional specific head index to visualize
        avg_heads: If True, average across all heads
    
    Returns:
        heat: 2D numpy array heatmap resized to orig_size
    """
    a = attn_tensor
    
    # Handle different attention tensor shapes
    # Possible shapes: [batch, heads, seq, seq] or [batch, seq, seq] or [heads, seq, seq]
    if a.dim() == 4:  # [batch, heads, seq, seq]
        if avg_heads:
            a = a.mean(1)  # Average across heads -> [batch, seq, seq]
        elif head_idx is not None:
            a = a[:, head_idx]  # Select specific head
        else:
            a = a.mean(1)  # Default to averaging
    
    if a.dim() == 3:  # [batch, seq, seq] or [heads, seq, seq]
        # Assume first dimension is batch if > 1, otherwise heads
        if a.shape[0] == 1:
            a = a[0]  # Remove batch dimension
        else:
            # Could be multiple heads, average them
            if not avg_heads and head_idx is not None and head_idx < a.shape[0]:
                a = a[head_idx]
            else:
                a = a.mean(0)
    
    # Now a should be [seq, seq]
    if a.dim() != 2:
        raise ValueError(f"Unexpected attention shape: {a.shape}")
    
    seq_len = a.shape[-1]
    
    # Extract CLS token attention (assuming CLS is first token)
    # CLS to all other tokens (patches): a[0, :]
    cls_to_patches = a[0, 1:]  # Skip CLS token itself, get patches
    
    n_patches = cls_to_patches.shape[0]
    
    # Try to reshape to square grid
    side = int(math.sqrt(n_patches))
    if side * side != n_patches:
        # If not perfect square, pad to next square
        side = int(math.ceil(math.sqrt(n_patches)))
        pad = side * side - n_patches
        cls_to_patches = torch.cat([cls_to_patches, torch.zeros(pad)], dim=0)
    
    # Reshape to 2D grid
    heat = cls_to_patches.reshape(side, side).numpy()
    
    # Normalize to [0, 1]
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    
    # Convert to uint8 for colormap
    heat = (heat * 255).astype(np.uint8)
    
    # Resize to original image size (width, height)
    heat = cv2.resize(heat, orig_size, interpolation=cv2.INTER_CUBIC)
    
    return heat


def overlay_heatmap(pil_img, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlay heatmap on original image.
    
    Args:
        pil_img: PIL Image
        heatmap: 2D numpy array (uint8) heatmap
        alpha: Transparency factor for overlay
        colormap: OpenCV colormap to use
    
    Returns:
        overlay: BGR image array with heatmap overlaid
    """
    # Convert PIL image to BGR (OpenCV format)
    img = np.array(pil_img)[:, :, ::-1].copy()  # RGB -> BGR
    
    # Apply colormap to heatmap
    colored = cv2.applyColorMap(heatmap, colormap)
    
    # Blend images
    overlay = cv2.addWeighted(img, 1 - alpha, colored, alpha, 0)
    
    return overlay


def visualize_attention_on_image(model, image_path, device='cuda', output_dir='attn_vis',
                                 attention_module_types=None, layer_idx=None, head_idx=None,
                                 avg_heads=True, save_raw=False):
    """
    Main function to visualize attention on a single image.
    
    Args:
        model: Trained model
        image_path: Path to input image
        device: Device to run on
        output_dir: Directory to save visualizations
        attention_module_types: Types of attention modules to hook
        layer_idx: Optional specific layer to visualize
        head_idx: Optional specific attention head to visualize
        avg_heads: If True, average across heads
        save_raw: If True, save raw attention maps as .npy files
    
    Returns:
        output_dir: Path to output directory
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model.to(device).eval()
    
    # Register hooks
    hooks, attn_maps, layer_names = register_attention_hooks(
        model, module_types=attention_module_types, layer_idx=layer_idx
    )
    
    if not hooks:
        print("Warning: No attention modules found to hook!")
        print("Make sure your model contains attention modules.")
        return None
    
    print(f"Registered {len(hooks)} hooks on attention modules")
    
    # Load image
    pil_img, tensor, orig_size = load_image(image_path, device)
    print(f"Loaded image: {image_path}, size: {orig_size}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run forward pass
    print("Running forward pass...")
    with torch.no_grad():
        try:
            # For caption models, we need to provide dummy sequence input
            # Just run the encoder part if possible
            if hasattr(model, 'encode'):
                _ = model.encode(tensor)
            elif hasattr(model, 'forward'):
                # Try running forward with minimal input
                _ = model(tensor)
            else:
                print("Could not determine how to run model forward pass")
                return None
        except Exception as e:
            print(f"Error during forward pass: {e}")
            print("This model may require additional inputs beyond just the image tensor.")
            return None
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Check if we captured any attention maps
    if not attn_maps:
        print("Error: No attention maps captured!")
        print("The model may not be storing attention weights correctly.")
        print("Try modifying the hook function to match your model's attention mechanism.")
        return None
    
    print(f"Captured {len(attn_maps)} attention maps")
    
    # Process and save each attention map
    for i, (att, layer_name) in enumerate(zip(attn_maps, layer_names)):
        print(f"\nProcessing attention map {i} from layer: {layer_name}")
        print(f"  Shape: {att.shape}")
        
        try:
            # Convert attention to heatmap
            heat = cls_attention_to_map(att, orig_size, head_idx=head_idx, avg_heads=avg_heads)
            
            # Create overlay
            overlay = overlay_heatmap(pil_img, heat, alpha=0.5)
            
            # Save overlay
            out_path = os.path.join(output_dir, f"attn_map_layer_{i}_{layer_name.replace('.', '_')}.png")
            cv2.imwrite(out_path, overlay)
            print(f"  Saved overlay: {out_path}")
            
            # Save raw heatmap
            heat_path = os.path.join(output_dir, f"heatmap_layer_{i}_{layer_name.replace('.', '_')}.png")
            cv2.imwrite(heat_path, cv2.applyColorMap(heat, cv2.COLORMAP_JET))
            print(f"  Saved heatmap: {heat_path}")
            
            # Optionally save raw numpy array
            if save_raw:
                npy_path = os.path.join(output_dir, f"attn_raw_layer_{i}.npy")
                np.save(npy_path, att.numpy())
                print(f"  Saved raw attention: {npy_path}")
                
        except Exception as e:
            print(f"  Error processing attention map {i}: {e}")
            continue
    
    print(f"\nVisualization complete! Results saved to: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description='Visualize attention weights on a single image',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with AoANet model
    python tools/visualize_attention.py \\
        --image path/to/image.jpg \\
        --infos_path log/infos_aoanet.pkl \\
        --model log/model.pth
    
    # Specify output directory and device
    python tools/visualize_attention.py \\
        --image data/sample.jpg \\
        --infos_path log/infos.pkl \\
        --model log/model.pth \\
        --output attn_vis \\
        --device cuda
    
    # Visualize specific layer and head
    python tools/visualize_attention.py \\
        --image data/sample.jpg \\
        --infos_path log/infos.pkl \\
        --model log/model.pth \\
        --layer 0 \\
        --head 2 \\
        --no-avg-heads
        """
    )
    
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint (.pth file)')
    parser.add_argument('--infos_path', type=str, required=True,
                       help='Path to infos pickle file')
    parser.add_argument('--output', type=str, default='attn_vis',
                       help='Output directory for visualizations (default: attn_vis)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use: cuda or cpu (default: cuda)')
    parser.add_argument('--layer', type=int, default=None,
                       help='Specific layer index to visualize (default: all layers)')
    parser.add_argument('--head', type=int, default=None,
                       help='Specific attention head to visualize (default: average all heads)')
    parser.add_argument('--no-avg-heads', action='store_true',
                       help='Do not average attention heads (requires --head)')
    parser.add_argument('--save-raw', action='store_true',
                       help='Save raw attention tensors as .npy files')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return 1
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model checkpoint not found: {args.model}")
        return 1
    
    # Check if infos exists
    if not os.path.exists(args.infos_path):
        print(f"Error: Infos file not found: {args.infos_path}")
        return 1
    
    # Load infos
    print(f"Loading infos from: {args.infos_path}")
    with open(args.infos_path, 'rb') as f:
        infos = utils.pickle_load(f)
    
    # Get model options
    opt = infos['opt']
    vocab = infos['vocab']
    
    # Setup model
    print(f"Setting up model: {opt.caption_model}")
    opt.vocab = vocab
    model = models.setup(opt)
    del opt.vocab
    
    # Load model weights
    print(f"Loading model weights from: {args.model}")
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    
    # Visualize attention
    avg_heads = not args.no_avg_heads
    output_dir = visualize_attention_on_image(
        model=model,
        image_path=args.image,
        device=args.device,
        output_dir=args.output,
        layer_idx=args.layer,
        head_idx=args.head,
        avg_heads=avg_heads,
        save_raw=args.save_raw
    )
    
    if output_dir is None:
        print("\nVisualization failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
