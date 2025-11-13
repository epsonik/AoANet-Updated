import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import skimage.transform


def visualize_attention_for_sequence(image_path, attention_weights, words, output_dir, att_size):
    """
    Visualizes attention weights for each word in a caption sequence.

    Args:
        image_path (str): Path to the original image.
        attention_weights (list of torch.Tensor): List of attention weight tensors.
        words (list of str): List of words in the caption.
        output_dir (str): Directory to save the visualizations.
        att_size (int): The width/height of the attention map (e.g., 7 for a 7x7 map).

    Returns:
        list: A list of paths to the saved visualization images.
    """
    image = Image.open(image_path)
    image = image.resize([256, 256], Image.LANCZOS)

    words = [word.replace('/', '') for word in words]

    vis_paths = []

    plt.figure(figsize=(20, 10))
    for t in range(len(words)):
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)

        # Get the attention weight and resize it
        current_alpha = attention_weights[t].cpu().numpy()
        alpha = skimage.transform.resize(current_alpha, [image.size[1], image.size[0]])

        if t > 0:  # Do not show heatmap for START token
            plt.imshow(alpha, alpha=0.7)

        plt.set_cmap(plt.cm.Greys_r)
        plt.axis('off')

    # Save the complete sequence visualization
    output_path = os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_att.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    vis_paths.append(output_path)

    return vis_paths


def capture_attention_weights(model, fc_feats, att_feats, att_masks, opt):
    """
    Generates a caption and then extracts attention weights for that caption.
    """
    model.eval()
    with torch.no_grad():
        # First, generate the sequence with _sample
        # model._sample returns (seq, seqLogprobs)
        seq, _ = model._sample(fc_feats, att_feats, att_masks, opt)

        # Then, use _forward to get attention weights for the generated sequence
        # model._forward returns (logit, state, att_weights)
        _, _, output_att = model._forward(fc_feats, att_feats, seq, att_masks)

    return seq, output_att


def get_attention_weights_from_sequence(model, fc_feats, att_feats, att_masks, seq):
    """
    This function is now the primary way to get attention, so capture_attention_weights will call it.
    The logic is kept here.
    """
    model.eval()
    with torch.no_grad():
        # Use _forward to get attention weights for a given sequence
        _, _, output_att = model._forward(fc_feats, att_feats, seq, att_masks)
    return output_att