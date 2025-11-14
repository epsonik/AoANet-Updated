"""
Demo script to showcase attention visualization without requiring a trained model.
This creates synthetic attention patterns to demonstrate the visualization capabilities.
"""
import numpy as np
import torch
from PIL import Image
import os
import matplotlib.pyplot as plt

import vis_utils


def create_demo_attention_patterns():
    """Create synthetic attention patterns for demonstration."""
    # Simulate attention for different words focusing on different regions
    
    # For a 7x7 grid (49 regions)
    att_size = 49
    grid_size = 7
    
    patterns = []
    
    # Pattern 1: Focus on top-left (object)
    attn = np.zeros(att_size)
    for i in range(2):
        for j in range(2):
            attn[i * grid_size + j] = 1.0
    attn = attn / attn.sum()
    patterns.append(attn)
    
    # Pattern 2: Focus on center (main object)
    attn = np.zeros(att_size)
    center = grid_size // 2
    for i in range(center - 1, center + 2):
        for j in range(center - 1, center + 2):
            attn[i * grid_size + j] = 1.0
    attn = attn / attn.sum()
    patterns.append(attn)
    
    # Pattern 3: Focus on right side (background object)
    attn = np.zeros(att_size)
    for i in range(grid_size):
        for j in range(5, 7):
            attn[i * grid_size + j] = 1.0
    attn = attn / attn.sum()
    patterns.append(attn)
    
    # Pattern 4: Focus on bottom (surface)
    attn = np.zeros(att_size)
    for i in range(5, 7):
        for j in range(grid_size):
            attn[i * grid_size + j] = 1.0
    attn = attn / attn.sum()
    patterns.append(attn)
    
    # Pattern 5: Distributed attention
    attn = np.ones(att_size) / att_size
    patterns.append(attn)
    
    return patterns


def main():
    """Run the demonstration."""
    print("=" * 60)
    print("Attention Visualization Demo")
    print("=" * 60)
    print()
    
    # Check if sample image exists
    sample_image = './vis/COCO_test2014_000000000027.jpg'
    if not os.path.exists(sample_image):
        print(f"Error: Sample image not found at {sample_image}")
        print("Please ensure the sample image exists.")
        return
    
    # Create output directory
    output_dir = './vis/demo_attention'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create synthetic attention patterns
    patterns = create_demo_attention_patterns()
    
    # Example caption words
    words = ['a', 'cat', 'sitting', 'on', 'couch']
    
    # Convert patterns to torch tensors (simulate model output)
    attention_weights = [torch.from_numpy(p).float().unsqueeze(0) for p in patterns]
    
    print(f"Image: {sample_image}")
    print(f"Caption: {' '.join(words)}")
    print(f"Number of words: {len(words)}")
    print(f"Attention regions per word: {len(patterns[0])}")
    print()
    
    # Create a subdirectory for this image based on its filename (without extension)
    image_basename = os.path.splitext(os.path.basename(sample_image))[0]
    image_output_dir = os.path.join(output_dir, image_basename)
    os.makedirs(image_output_dir, exist_ok=True)
    
    # Create visualizations
    print("Generating visualizations...")
    vis_paths = vis_utils.visualize_attention_for_sequence(
        sample_image,
        attention_weights,
        words,
        output_dir=image_output_dir,
        att_size=49
    )
    
    print(f"\nGenerated {len(vis_paths)} visualization(s)")
    print(f"Output directory: {image_output_dir}")
    print()
    
    # List generated files
    print("Generated files:")
    for path in vis_paths:
        print(f"  - {os.path.basename(path)}")
    
    summary_path = os.path.join(image_output_dir, 'COCO_test2014_000000000027_summary.png')
    if os.path.exists(summary_path):
        print(f"  - {os.path.basename(summary_path)}")
    
    print()
    print("=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print()
    print("The visualizations show synthetic attention patterns:")
    print("  - 'a': Focus on top-left region")
    print("  - 'cat': Focus on center (main object)")
    print("  - 'sitting': Focus on right side")
    print("  - 'on': Focus on bottom (surface)")
    print("  - 'couch': Distributed attention")
    print()
    print("In a real model, these would correspond to actual attention")
    print("computed by the neural network during caption generation.")


if __name__ == '__main__':
    main()
