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

    attention_weights = []

    # Define a hook to capture the attention weights from the module's state
    def hook(module, input, output):
        # The attention weights are stored in module.attn
        if hasattr(module, 'attn'):
            weights = module.attn.data.cpu().numpy()
            # Handle different shapes during beam search vs. greedy
            if weights.ndim > 2:
                # Average over the attention heads (axis=1)
                # This gives us (batch*beam, query_len, key_len)
                attention_weights.append(np.mean(weights, axis=1))

    hook_handle = model.core.attention.register_forward_hook(hook)

    # Generate sequence
    seq, _ = model.sample(fc_feats, att_feats, att_masks, opt)

    hook_handle.remove()

    if attention_weights:
        # For beam search, we only need the weights for the top beam (index 0).
        num_steps = len(attention_weights)
        seq_len = seq.size(1)

        # Take the attention from the top beam at each step
        processed_weights = [att_step[0] for att_step in attention_weights]

        # The number of attention steps can be > seq_len, so truncate.
        final_weights = np.array(processed_weights[:seq_len])
        return seq, final_weights
    else:
        return seq, []


def get_attention_weights_from_sequence(model, fc_feats, att_feats, att_masks, seq):
    """
    Re-runs the model with a given sequence to extract attention weights.
    """
    model.eval()

    batch_size = fc_feats.size(0)
    assert batch_size == 1, "Currently only supports batch size of 1"

    wt = fc_feats.new_zeros(batch_size, seq.size(1), dtype=torch.long)
    wt[:, 0] = model.bos_idx
    wt[:, 1:] = seq[:, :-1]

    attention_weights = []

    def hook(module, input, output):
        if hasattr(module, 'attn'):
            weights = module.attn.data.cpu().numpy()
            if weights.ndim > 2:
                attention_weights.append(np.mean(weights, axis=1))

    hook_handle = model.core.attention.register_forward_hook(hook)

    _ = model(fc_feats, att_feats, wt, att_masks)

    hook_handle.remove()

    if attention_weights:
        # Squeeze out the batch dimension
        return np.array(attention_weights).squeeze()
    else:
        return []

# ----------------- VISUALIZATION UTILS -----------------

def resize_attention_to_image(attention, img_size, grid_size):
    """
    Resizes the attention map to the original image size.
    """
    # Reshape from 1D to 2D grid
    attention_grid = attention.reshape(grid_size, grid_size)

    # Resize to image size
    attention_map = cv2.resize(attention_grid, (img_size[0], img_size[1]),
                               interpolation=cv2.INTER_LINEAR)
    return attention_map

def create_heatmap(attention_map, image):
    """
    Creates a heatmap overlay on the image.
    """
    # Normalize attention map for heatmap
    heatmap = (attention_map - np.min(attention_map)) / (np.max(attention_map) - np.min(attention_map) + 1e-8)
    heatmap = (heatmap * 255).astype(np.uint8)

    # Apply a color map
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Blend the heatmap with the original image
    overlay = cv2.addWeighted(image, 0.6, colored_heatmap, 0.4, 0)
    return overlay

def add_caption_to_image(image, caption, word, font_path=None, font_size=20):
    """
    Adds the caption and highlights the current word on the image.
    """
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    display_words = [(w, w == word) for w in caption.split()]

    x, y = 10, 10
    for w, is_highlighted in display_words:
        try: # Use textbbox for more accurate size
            bbox = draw.textbbox((x, y), w, font=font)
        except AttributeError: # Fallback for older Pillow
            text_size = draw.textsize(w, font=font)
            bbox = (x, y, x + text_size[0], y + text_size[1])

        if is_highlighted:
            draw.rectangle(bbox, fill="yellow")
            draw.text((x, y), w, font=font, fill="black")
        else:
            draw.text((x, y), w, font=font, fill="white")

        x = bbox[2] + 5 # Move x for the next word

    return image


def visualize_attention_for_sequence(image_path, attention_weights, words, output_dir, att_size):
    """
    Creates and saves a sequence of visualizations for each word in the caption.
    """
    img = Image.open(image_path).convert("RGB")
    # --- POCZĄTEK ZMIANY ---
    # Skalowanie obrazu do rozmiaru 336x336
    img = img.resize([14 * 24, 14 * 24], Image.LANCZOS)
    # --- KONIEC ZMIANY ---
    img_cv = np.array(img)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    # Calculate the grid size (e.g., 14 for a 14x14 grid)
    grid_size = int(np.sqrt(att_size))

    vis_paths = []

    for i, (word, attention) in enumerate(zip(words, attention_weights)):
        # The attention is over the input tokens (text + image). We want the image part.
        # Shape is (query_len, key_len). e.g., (1, 197) where 196 is image regions
        vis_attention = attention[0, -att_size:]
        
        # Convert to numpy if it's a tensor
        if isinstance(vis_attention, torch.Tensor):
            vis_attention = vis_attention.cpu().numpy()

        # Create the heatmap
        attention_map = resize_attention_to_image(vis_attention, (img.width, img.height), grid_size)
        heatmap_overlay = create_heatmap(attention_map, img_cv)

        # Convert back to PIL
        vis_image = Image.fromarray(cv2.cvtColor(heatmap_overlay, cv2.COLOR_BGR2RGB))

        # The line to add the caption has been removed as requested.

        # Save the visualization with the word in the filename
        sanitized_word = "".join(c for c in word if c.isalnum() or c in (' ', '_')).rstrip()
        filename = f"{i}_{sanitized_word}.png"
        output_path = os.path.join(output_dir, filename)
        vis_image.save(output_path)
        vis_paths.append(output_path)

    return vis_paths