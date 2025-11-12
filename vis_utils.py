"""
Utilities for visualizing attention weights in image captioning models.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import cv2

# ----------------- ATTENTION CAPTURING UTILS -----------------

def capture_attention_weights(model, fc_feats, att_feats, att_masks, opt):
    """
    Runs the model in evaluation mode to generate a caption and capture attention weights.
    """
    model.eval()

    # Store weights in a list
    attention_weights = []

    # Define a hook to capture the output of the AoA module
    def hook(module, input, output):
        # output[1] is the attention weights from AoA
        # The output shape is (batch_size, num_heads, seq_len, seq_len)
        # For image features, seq_len is the number of regions (e.g., 196)
        # We take the mean over the heads to get (batch_size, seq_len, seq_len)
        # For visualization, we are interested in the attention over the image regions,
        # which corresponds to the last dimension of the attention map.
        # We take the mean across the second-to-last dimension.
        weights = output[1].data.cpu().numpy()
        attention_weights.append(np.mean(weights, axis=(1, 2)))

    # Register the hook on the AoA module of the attention mechanism
    hook_handle = model.att.aoa_layer.register_forward_hook(hook)

    # Generate sequence
    seq, _ = model.sample(fc_feats, att_feats, att_masks, opt)

    # Remove the hook
    hook_handle.remove()

    # The attention_weights list will contain arrays for each decoding step.
    # We need to rearrange them to be per-sequence-step.
    # If seq has shape (1, max_len), and we have max_len steps,
    # attention_weights will have max_len arrays of shape (1, num_regions)
    # We squeeze and stack them.
    if attention_weights:
        # Stack the weights from each step and transpose
        # to get (seq_len, num_regions)
        return seq, np.array(attention_weights).squeeze(axis=1)
    else:
        return seq, []


def get_attention_weights_from_sequence(model, fc_feats, att_feats, att_masks, seq):
    """
    Re-runs the model with a given sequence to extract attention weights.
    This is useful when beam search is used for generation, as hooks might not capture
    the attention for the final selected sequence.
    """
    model.eval()

    batch_size = fc_feats.size(0)
    assert batch_size == 1, "Currently only supports batch size of 1"

    # Prepare input for the model
    wt = fc_feats.new_zeros(batch_size, seq.size(1), dtype=torch.long)
    wt[:, 0] = model.bos_idx
    wt[:, 1:] = seq[:, :-1]

    attention_weights = []

    def hook(module, input, output):
        weights = output[1].data.cpu().numpy()
        attention_weights.append(np.mean(weights, axis=(1, 2)))

    hook_handle = model.att.aoa_layer.register_forward_hook(hook)

    # Forward pass with the given sequence
    _ = model(fc_feats, att_feats, wt, att_masks)

    hook_handle.remove()

    if attention_weights:
        return np.array(attention_weights).squeeze(axis=1)
    else:
        return []

# ----------------- VISUALIZATION UTILS -----------------

def resize_attention_to_image(attention, img_size, grid_size):
    """
    Resizes the attention map to the original image size.
    """
    # Reshape attention to a square grid
    attention_grid = attention.reshape(grid_size, grid_size)

    # Resize to image size
    attention_map = cv2.resize(attention_grid, (img_size[0], img_size[1]),
                               interpolation=cv2.INTER_LINEAR)
    return attention_map

def create_heatmap(attention_map, image):
    """
    Creates a heatmap overlay on the image.
    """
    # Normalize attention map
    heatmap = (attention_map - np.min(attention_map)) / (np.max(attention_map) - np.min(attention_map))
    heatmap = (heatmap * 255).astype(np.uint8)

    # Apply colormap
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay on image
    overlay = cv2.addWeighted(image, 0.6, colored_heatmap, 0.4, 0)
    return overlay

def add_caption_to_image(image, caption, word, font_path=None, font_size=20):
    """
    Adds the caption and highlights the current word on the image.
    """
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except (IOError, TypeError):
        font = ImageFont.load_default()

    # Create a list of (word, is_highlighted)
    display_words = [(w, w == word) for w in caption.split()]

    x, y = 10, 10
    for w, is_highlighted in display_words:
        text_size = draw.textsize(w, font=font)
        if is_highlighted:
            draw.rectangle([x, y, x + text_size[0], y + text_size[1]], fill="yellow")
            draw.text((x, y), w, font=font, fill="black")
        else:
            draw.text((x, y), w, font=font, fill="white")
        x += text_size[0] + 5

    return image


def visualize_attention_for_sequence(image_path, attention_weights, words, output_dir, att_size):
    """
    Creates and saves a sequence of visualizations for each word in the caption.
    """
    img = Image.open(image_path).convert("RGB")
    img_cv = np.array(img)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    grid_size = int(np.sqrt(att_size))

    vis_paths = []

    # Add a visualization for the <start> token (no highlight)
    # You might want to visualize the initial attention state

    for i, (word, attention) in enumerate(zip(words, attention_weights)):
        # Resize attention and create heatmap
        attention_map = resize_attention_to_image(attention, (img.width, img.height), grid_size)
        heatmap_overlay = create_heatmap(attention_map, img_cv)

        # Convert back to PIL Image
        vis_image = Image.fromarray(cv2.cvtColor(heatmap_overlay, cv2.COLOR_BGR2RGB))

        # Add caption with highlighted word
        vis_image = add_caption_to_image(vis_image, " ".join(words), word)

        # Save visualization
        filename = f"{os.path.basename(image_path).split('.')[0]}_step_{i}_{word}.png"
        save_path = os.path.join(output_dir, filename)
        vis_image.save(save_path)
        vis_paths.append(save_path)

    return vis_paths
