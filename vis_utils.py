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
import skimage.transform


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


def resize_attention_to_image(attention, image_shape, att_size, smooth=True):
    """
    Resize attention weights to match the image dimensions.
    
    Args:
        attention: Attention weights (att_size,) numpy array
        image_shape: Target image shape (height, width)
        att_size: Number of attention regions (e.g., 49 for 7x7 grid)
        smooth: If True, use smooth upscaling with pyramid_expand
    
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
    
    # Calculate upscale factor to match target image size
    upscale = image_shape[0] // grid_size
    
    if smooth and upscale > 1:
        # Use smooth upscaling like reference implementation
        attention_resized = skimage.transform.pyramid_expand(
            attention_grid, upscale=upscale, sigma=8
        )
        # Crop or pad to exact size if needed
        if attention_resized.shape[0] != image_shape[0] or attention_resized.shape[1] != image_shape[1]:
            attention_resized = skimage.transform.resize(
                attention_resized, image_shape, mode='reflect', anti_aliasing=True
            )
    else:
        # Use standard resizing
        attention_resized = skimage.transform.resize(
            attention_grid, image_shape, mode='reflect', anti_aliasing=True
        )
    
    # Normalize to [0, 1]
    if attention_resized.max() > attention_resized.min():
        attention_resized = (attention_resized - attention_resized.min()) / (
            attention_resized.max() - attention_resized.min()
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
                                     output_dir='vis/attention', att_size=49, smooth=True):
    """
    Create attention visualizations for each word in the generated sequence.
    Similar to the reference implementation with overlay style.
    
    Args:
        image_path: Path to the original image
        attention_weights: List of attention weight arrays, one per word
        word_sequence: List of words in the generated caption
        output_dir: Directory to save visualizations
        att_size: Number of attention regions
        smooth: If True, use smooth upscaling for attention maps
    
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
    
    # Save original image
    image.save(os.path.join(output_dir, 'original.png'))
    
    visualization_paths = []
    
    num_words = min(len(attention_weights), len(word_sequence))
    
    if num_words == 0:
        return []
    
    # Create individual visualizations (overlay style like reference)
    for t, (attention, word) in enumerate(zip(attention_weights, word_sequence)):
        if isinstance(attention, torch.Tensor):
            attention = attention.numpy()
        
        # Take first batch element if batched
        if attention.ndim > 1:
            attention = attention[0]
        
        # Resize attention to image dimensions with smooth upscaling
        attention_map = resize_attention_to_image(
            attention, image_np.shape[:2], att_size, smooth=smooth
        )
        
        # Create figure with just the overlay (like reference implementation)
        fig, ax = plt.subplots()
        ax.imshow(image_np)
        
        # Overlay attention with jet colormap and alpha=0.6 (like reference)
        ax.imshow(attention_map, alpha=0.6, cmap='jet')
        ax.axis('off')
        
        # Sanitize word for filename
        sanitized_word = "".join(c for c in word if c.isalnum() or c in (' ', '_')).rstrip()
        output_path = os.path.join(output_dir, f'{t}_{sanitized_word}.png')
        
        # Save with tight bbox and no padding (like reference)
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
        visualization_paths.append(output_path)
    
    # Create a summary visualization with all words
    create_summary_visualization(
        image_np, attention_weights, word_sequence, 
        os.path.join(output_dir, f'{base_name}_summary.png'),
        att_size, smooth=smooth
    )
    
    return visualization_paths


def create_summary_visualization(image, attention_weights, word_sequence, 
                                output_path, att_size=49, smooth=True):
    """
    Create a single figure showing attention for all words in a grid.
    Similar to reference implementation with horizontal layout.
    
    Args:
        image: Original image as numpy array
        attention_weights: List of attention weight arrays
        word_sequence: List of words
        output_path: Path to save the summary visualization
        att_size: Number of attention regions
        smooth: If True, use smooth upscaling for attention maps
    """
    num_words = min(len(attention_weights), len(word_sequence), 50)  # Limit to 50 words
    
    if num_words == 0:
        return
    
    # Create figure with horizontal layout (like reference)
    subplot_size = 4
    num_col = num_words
    fig = plt.figure(dpi=100)
    fig.set_size_inches(subplot_size * num_col, subplot_size * 4)
    
    img_size = 4
    fig_height = img_size
    fig_width = num_col + img_size
    
    # Use GridSpec for flexible layout
    grid = plt.GridSpec(fig_height, fig_width)
    
    # Show original image on the left
    ax_orig = plt.subplot(grid[0:img_size, 0:img_size])
    ax_orig.imshow(image)
    ax_orig.axis('off')
    
    # Show attention for each word
    for t in range(num_words):
        attention = attention_weights[t]
        word = word_sequence[t]
        
        if isinstance(attention, torch.Tensor):
            attention = attention.numpy()
        
        if attention.ndim > 1:
            attention = attention[0]
        
        # Resize attention with smooth upscaling
        attention_map = resize_attention_to_image(
            attention, image.shape[:2], att_size, smooth=smooth
        )
        
        # Create subplot for this word
        ax = plt.subplot(grid[fig_height - 1, img_size + t])
        ax.imshow(image)
        ax.imshow(attention_map, alpha=0.6, cmap='jet')
        ax.axis('off')
    
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    
    print(f'Summary visualization saved to: {output_path}')
