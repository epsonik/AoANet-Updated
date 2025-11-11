"""
Visualization utilities for attention weights in image captioning models.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class AttentionHook:
    """Hook to capture attention weights during forward pass."""
    def __init__(self):
        self.attention_weights = []
        
    def __call__(self, module, input, output):
        """Capture attention weights from the module."""
        if hasattr(module, 'attn') and module.attn is not None:
            # Multi-headed attention
            attn = module.attn.mean(dim=1)  # Average across heads
            self.attention_weights.append(attn.detach().cpu())
            
    def reset(self):
        """Clear stored attention weights."""
        self.attention_weights = []


def capture_attention_weights(model, fc_feats, att_feats, att_masks, opt={}):
    """
    Generate a caption and capture attention weights for each decoding step.
    
    Args:
        model: The caption model
        fc_feats: Image features (batch_size, fc_feat_size)
        att_feats: Attention features (batch_size, att_size, att_feat_size)
        att_masks: Attention masks (batch_size, att_size) or None
        opt: Options for sampling (beam_size, etc.)
    
    Returns:
        seq: Generated sequence (batch_size, seq_length)
        attention_weights: List of attention weight tensors, one per timestep
    """
    model.eval()
    
    # Register hooks to capture attention
    attention_hook = AttentionHook()
    hooks = []
    
    # Register hook on attention modules
    if hasattr(model, 'core') and hasattr(model.core, 'attention'):
        hook = model.core.attention.register_forward_hook(attention_hook)
        hooks.append(hook)
    
    # If model has refiner with attention (AoA model)
    if hasattr(model, 'refiner'):
        for layer in model.refiner.layers if hasattr(model.refiner, 'layers') else []:
            if hasattr(layer, 'self_attn'):
                hook = layer.self_attn.register_forward_hook(attention_hook)
                hooks.append(hook)
    
    try:
        with torch.no_grad():
            # Generate caption
            seq, seqLogprobs = model(fc_feats, att_feats, att_masks, opt=opt, mode='sample')
        
        attention_weights = attention_hook.attention_weights
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()
    
    return seq, attention_weights


def get_attention_weights_from_sequence(model, fc_feats, att_feats, att_masks, seq):
    """
    Extract attention weights by re-running the model with a given sequence.
    This is useful when you already have a generated caption and want to visualize attention.
    
    Args:
        model: The caption model
        fc_feats: Image features (batch_size, fc_feat_size)
        att_feats: Attention features (batch_size, att_size, att_feat_size)
        att_masks: Attention masks (batch_size, att_size) or None
        seq: Pre-generated sequence (batch_size, seq_length)
    
    Returns:
        attention_weights: List of attention weight tensors, one per timestep
    """
    model.eval()
    batch_size = fc_feats.size(0)
    state = model.init_hidden(batch_size)
    
    # Prepare features
    p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = model._prepare_feature(
        fc_feats, att_feats, att_masks)
    
    attention_weights = []
    
    with torch.no_grad():
        for t in range(seq.size(1)):
            if seq[0, t] == 0:  # Stop at end token
                break
                
            it = seq[:, t].clone()
            
            # Store attention before forward pass
            attention_module = None
            if hasattr(model, 'core') and hasattr(model.core, 'attention'):
                attention_module = model.core.attention
            
            # Forward pass
            logprobs, state = model.get_logprobs_state(
                it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)
            
            # Extract attention weights
            if attention_module is not None:
                if hasattr(attention_module, 'attn') and attention_module.attn is not None:
                    # Multi-headed attention (averaged across heads)
                    attn = attention_module.attn.mean(dim=1)  # (batch, att_size)
                    attention_weights.append(attn.cpu())
            
    return attention_weights


def resize_attention_to_image(attention, image_shape, att_size):
    """
    Resize attention weights to match the image dimensions.
    
    Args:
        attention: Attention weights (att_size,) numpy array
        image_shape: Target image shape (height, width)
        att_size: Number of attention regions (e.g., 49 for 7x7 grid)
    
    Returns:
        attention_map: 2D attention map resized to image_shape
    """
    # Reshape attention to spatial grid (assuming square grid)
    grid_size = int(np.sqrt(att_size))
    if grid_size * grid_size != att_size:
        # If not a perfect square, handle gracefully
        # This might be the case for bottom-up features
        # For now, we'll create a square representation
        grid_size = int(np.ceil(np.sqrt(att_size)))
        attention_padded = np.zeros(grid_size * grid_size)
        attention_padded[:len(attention)] = attention
        attention = attention_padded
    
    attention_grid = attention.reshape(grid_size, grid_size)
    
    # Resize to image dimensions
    attention_resized = cv2.resize(
        attention_grid,
        (image_shape[1], image_shape[0]),
        interpolation=cv2.INTER_CUBIC
    )
    
    # Normalize to [0, 1]
    attention_resized = (attention_resized - attention_resized.min()) / (
        attention_resized.max() - attention_resized.min() + 1e-8
    )
    
    return attention_resized


def create_attention_heatmap(image, attention_map, alpha=0.6, colormap=cv2.COLORMAP_JET):
    """
    Create a heatmap visualization by overlaying attention on the image.
    
    Args:
        image: Original image as numpy array (H, W, 3) in RGB format
        attention_map: 2D attention map (H, W) normalized to [0, 1]
        alpha: Blending factor for overlay (0 = only image, 1 = only heatmap)
        colormap: OpenCV colormap for heatmap visualization
    
    Returns:
        heatmap: Blended image with attention heatmap
    """
    # Convert image to uint8 if needed
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Convert attention map to heatmap
    attention_map_uint8 = (attention_map * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(attention_map_uint8, colormap)
    
    # Convert heatmap from BGR to RGB (OpenCV uses BGR)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Blend image and heatmap
    blended = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    
    return blended


def visualize_attention_for_sequence(image_path, attention_weights, word_sequence, 
                                     output_dir='vis/attention', att_size=49):
    """
    Create attention visualizations for each word in the generated sequence.
    
    Args:
        image_path: Path to the original image
        attention_weights: List of attention weight arrays, one per word
        word_sequence: List of words in the generated caption
        output_dir: Directory to save visualizations
        att_size: Number of attention regions
    
    Returns:
        visualization_paths: List of paths to saved visualization images
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    visualization_paths = []
    
    # Create a figure with subplots for all words
    num_words = min(len(attention_weights), len(word_sequence))
    
    if num_words == 0:
        return []
    
    # Create individual visualizations
    for t, (attention, word) in enumerate(zip(attention_weights, word_sequence)):
        if isinstance(attention, torch.Tensor):
            attention = attention.numpy()
        
        # Take first batch element if batched
        if attention.ndim > 1:
            attention = attention[0]
        
        # Resize attention to image dimensions
        attention_map = resize_attention_to_image(
            attention, image_np.shape[:2], att_size
        )
        
        # Create heatmap
        heatmap = create_attention_heatmap(image_np, attention_map)
        
        # Create figure with original image and heatmap side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original image
        ax1.imshow(image_np)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Heatmap
        ax2.imshow(heatmap)
        ax2.set_title(f'Attention for: "{word}"')
        ax2.axis('off')
        
        # Save figure
        output_path = os.path.join(output_dir, f'{base_name}_word_{t:02d}_{word}.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        visualization_paths.append(output_path)
    
    # Create a summary visualization with all words
    create_summary_visualization(
        image_np, attention_weights, word_sequence, 
        os.path.join(output_dir, f'{base_name}_summary.png'),
        att_size
    )
    
    return visualization_paths


def create_summary_visualization(image, attention_weights, word_sequence, 
                                output_path, att_size=49):
    """
    Create a single figure showing attention for all words in a grid.
    
    Args:
        image: Original image as numpy array
        attention_weights: List of attention weight arrays
        word_sequence: List of words
        output_path: Path to save the summary visualization
        att_size: Number of attention regions
    """
    num_words = min(len(attention_weights), len(word_sequence), 12)  # Limit to 12 words
    
    if num_words == 0:
        return
    
    # Calculate grid dimensions
    cols = min(4, num_words)
    rows = (num_words + cols - 1) // cols
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)
    
    for idx in range(num_words):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        attention = attention_weights[idx]
        word = word_sequence[idx]
        
        if isinstance(attention, torch.Tensor):
            attention = attention.numpy()
        
        if attention.ndim > 1:
            attention = attention[0]
        
        # Resize and create heatmap
        attention_map = resize_attention_to_image(attention, image.shape[:2], att_size)
        heatmap = create_attention_heatmap(image, attention_map)
        
        ax.imshow(heatmap)
        ax.set_title(f'"{word}"', fontsize=10)
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(num_words, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'Summary visualization saved to: {output_path}')
