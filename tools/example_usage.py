#!/usr/bin/env python
"""
Example Python script showing how to use the attention visualization functions.

This demonstrates how to integrate attention visualization into your own scripts
or notebooks, rather than using the command-line interface.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import models
import misc.utils as utils
from tools.visualize_attention import visualize_attention_on_image


def example_basic():
    """Example 1: Basic usage - load model and visualize attention."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Update these paths to match your setup
    model_path = "log/log_aoanet_rl/model.pth"
    infos_path = "log/log_aoanet_rl/infos_aoanet.pkl"
    image_path = "data/sample_image.jpg"
    
    # Check if files exist
    if not all(os.path.exists(p) for p in [model_path, infos_path, image_path]):
        print("Skipping - required files not found")
        print(f"  Model: {model_path}")
        print(f"  Infos: {infos_path}")
        print(f"  Image: {image_path}")
        return
    
    # Load model
    print("\n1. Loading model...")
    with open(infos_path, 'rb') as f:
        infos = utils.pickle_load(f)
    
    opt = infos['opt']
    vocab = infos['vocab']
    
    opt.vocab = vocab
    model = models.setup(opt)
    del opt.vocab
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    print("   Model loaded successfully!")
    
    # Visualize attention
    print("\n2. Visualizing attention...")
    output_dir = visualize_attention_on_image(
        model=model,
        image_path=image_path,
        device='cpu',  # Use CPU for this example
        output_dir='example_output_basic'
    )
    
    if output_dir:
        print(f"\n✓ Success! Visualizations saved to: {output_dir}")
    else:
        print("\n✗ Visualization failed")


def example_specific_layer():
    """Example 2: Visualize specific layer only."""
    print("\n" + "=" * 60)
    print("Example 2: Visualize Specific Layer")
    print("=" * 60)
    
    # This would be similar to example_basic but with layer_idx parameter
    print("\nTo visualize a specific layer, pass layer_idx parameter:")
    print("""
    output_dir = visualize_attention_on_image(
        model=model,
        image_path=image_path,
        device='cpu',
        output_dir='example_output_layer0',
        layer_idx=0  # Only visualize layer 0
    )
    """)


def example_specific_head():
    """Example 3: Visualize specific attention head."""
    print("\n" + "=" * 60)
    print("Example 3: Visualize Specific Attention Head")
    print("=" * 60)
    
    print("\nTo visualize a specific attention head:")
    print("""
    output_dir = visualize_attention_on_image(
        model=model,
        image_path=image_path,
        device='cpu',
        output_dir='example_output_head2',
        head_idx=2,        # Visualize head 2
        avg_heads=False    # Don't average heads
    )
    """)


def example_with_raw_save():
    """Example 4: Save raw attention tensors."""
    print("\n" + "=" * 60)
    print("Example 4: Save Raw Attention Tensors")
    print("=" * 60)
    
    print("\nTo save raw attention tensors as .npy files:")
    print("""
    output_dir = visualize_attention_on_image(
        model=model,
        image_path=image_path,
        device='cpu',
        output_dir='example_output_raw',
        save_raw=True  # Save raw attention as .npy
    )
    """)


def example_custom_hooks():
    """Example 5: Custom attention module types."""
    print("\n" + "=" * 60)
    print("Example 5: Custom Attention Module Types")
    print("=" * 60)
    
    print("\nIf your model uses custom attention modules:")
    print("""
    from mymodel import CustomAttentionModule
    
    output_dir = visualize_attention_on_image(
        model=model,
        image_path=image_path,
        device='cpu',
        output_dir='example_output_custom',
        attention_module_types=(CustomAttentionModule,)
    )
    """)


def example_batch_processing():
    """Example 6: Process multiple images."""
    print("\n" + "=" * 60)
    print("Example 6: Batch Processing Multiple Images")
    print("=" * 60)
    
    print("\nTo process multiple images:")
    print("""
    import glob
    
    # Load model once
    model = ...  # Load your model
    
    # Process all images in a directory
    for image_path in glob.glob('data/images/*.jpg'):
        basename = os.path.basename(image_path).replace('.jpg', '')
        output_dir = f'visualizations/{basename}'
        
        visualize_attention_on_image(
            model=model,
            image_path=image_path,
            device='cuda',
            output_dir=output_dir
        )
        
        print(f'Processed: {image_path}')
    """)


def example_analyze_attention():
    """Example 7: Analyze attention patterns programmatically."""
    print("\n" + "=" * 60)
    print("Example 7: Analyze Attention Patterns")
    print("=" * 60)
    
    print("\nFor programmatic analysis of attention:")
    print("""
    from tools.visualize_attention import (
        register_attention_hooks, load_image, cls_attention_to_map
    )
    import numpy as np
    
    # Register hooks
    hooks, attn_maps, layer_names = register_attention_hooks(model)
    
    # Load and process image
    pil_img, tensor, orig_size = load_image(image_path, device='cpu')
    
    # Run forward pass
    with torch.no_grad():
        _ = model(tensor)
    
    # Clean up hooks
    for h in hooks:
        h.remove()
    
    # Analyze attention patterns
    for i, attn in enumerate(attn_maps):
        # Get attention map
        heat = cls_attention_to_map(attn, orig_size)
        
        # Analyze statistics
        mean_attn = heat.mean()
        max_attn = heat.max()
        std_attn = heat.std()
        
        print(f'Layer {i}:')
        print(f'  Mean attention: {mean_attn:.2f}')
        print(f'  Max attention: {max_attn:.2f}')
        print(f'  Std attention: {std_attn:.2f}')
        
        # Find regions with high attention
        threshold = mean_attn + std_attn
        high_attn_mask = heat > threshold
        high_attn_ratio = high_attn_mask.sum() / heat.size
        
        print(f'  High attention ratio: {high_attn_ratio:.2%}')
    """)


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("ATTENTION VISUALIZATION - Python Examples")
    print("=" * 70)
    
    # Only run example 1 if files exist, others are just documentation
    example_basic()
    
    # Print other examples (documentation)
    example_specific_layer()
    example_specific_head()
    example_with_raw_save()
    example_custom_hooks()
    example_batch_processing()
    example_analyze_attention()
    
    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)
    print("\nNote: Most examples above are for documentation.")
    print("Update the file paths in example_basic() to run actual visualizations.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
