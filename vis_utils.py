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

    def hook(module, input, output):
        # The 'output' of the attention module is a tuple where the second element
        # is the attention weights.
        if isinstance(output, tuple) and len(output) > 1:
            weights = output[1].data.cpu().numpy()
            # During beam search, the batch dimension may be expanded.
            # We want to handle weights of shape (batch*beam, heads, len, len)
            # or (batch, heads, len, len)
            if weights.ndim > 2:
                # Average over the attention heads.
                attention_weights.append(np.mean(weights, axis=1))

    hook_handle = model.core.attention.register_forward_hook(hook)

    # Generate sequence
    seq, _ = model.sample(fc_feats, att_feats, att_masks, opt)

    hook_handle.remove()

    if attention_weights:
        # For beam search, we get weights for each beam. We only need the ones for the top beam.
        # The number of steps is len(attention_weights).
        # The first dimension of each item is batch_size * beam_size.
        # We take the first one of each step.
        num_steps = len(attention_weights)
        seq_len = seq.size(1)

        # Take the attention from the top beam (index 0) at each step
        processed_weights = [att_step[0] for att_step in attention_weights]

        # We might have more attention steps than words in the final sequence
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
        if isinstance(output, tuple) and len(output) > 1:
            weights = output[1].data.cpu().numpy()
            if weights.ndim > 2:
                attention_weights.append(np.mean(weights, axis=1))


    hook_handle = model.core.attention.register_forward_hook(hook)

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
    attention_grid = attention.reshape(grid_size, grid_size)

    attention_map = cv2.resize(attention_grid, (img_size[0], img_size[1]),
                               interpolation=cv2.INTER_LINEAR)
    return attention_map

def create_heatmap(attention_map, image):
    """
    Creates a heatmap overlay on the image.
    """
    heatmap = (attention_map - np.min(attention_map)) / (np.max(attention_map) - np.min(attention_map) + 1e-8)
    heatmap = (heatmap * 255).astype(np.uint8)

    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

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

    display_words = [(w, w == word) for w in caption.split()]

    x, y = 10, 10
    for w, is_highlighted in display_words:
        # Use textbbox for more accurate size calculation
        try:
            bbox = draw.textbbox((x, y), w, font=font)
        except AttributeError: # Fallback for older Pillow versions
            bbox = draw.textsize(w, font=font)
            bbox = (x, y, x + bbox[0], y + bbox[1])


        if is_highlighted:
            draw.rectangle(bbox, fill="yellow")
            draw.text((x, y), w, font=font, fill="black")
        else:
            draw.text((x, y), w, font=font, fill="white")

        x = bbox[2] + 5


    return image


def visualize_attention_for_sequence(image_path, attention_weights, words, output_dir, att_size):
    """
    Creates and saves a sequence of visualizations for each word in the caption.
    """
    img = Image.open(image_path).convert("RGB")
    img_cv = np.array(img)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    # The attention weights we get are over the input tokens.
    # We are interested in the attention over the visual features.
    # This corresponds to the last elements of the attention vector.

    vis_paths = []

    for i, (word, attention) in enumerate(zip(words, attention_weights)):
        # attention shape is (query_len, key_len), e.g. (1, 197)
        # We take the last `att_size` elements, which correspond to the image features.
        vis_attention = attention[0, -att_size:]
        grid_size = int(np.sqrt(vis_attention.shape[0]))


        attention_map = resize_attention_to_image(vis_attention, (img.width, img.height), grid_size)
        heatmap_overlay = create_heatmap(attention_map, img_cv)

        vis_image = Image.fromarray(cv2.cvtColor(heatmap_overlay, cv2.COLOR_BGR2RGB))

        vis_image = add_caption_to_image(vis_image, " ".join(words), word)

        filename = f"{os.path.basename(image_path).split('.')[0]}_step_{i}_{word}.png"
        save_path = os.path.join(output_dir, filename)
        vis_image.save(save_path)
        vis_paths.append(save_path)

    return vis_paths