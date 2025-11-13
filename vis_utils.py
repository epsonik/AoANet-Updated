import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import skimage.transform


def visualize_attention_for_sequence(image_path, attention_weights, words, output_dir, att_size):
    """
    Visualizes attention weights for each word in a caption sequence.
    """
    image = Image.open(image_path)
    image = image.resize([256, 256], Image.LANCZOS)

    words = [word.replace('/', '') for word in words]

    vis_paths = []

    plt.figure(figsize=(20, 10))
    # Determine grid size
    num_words = len(words)
    grid_size = int(np.ceil(np.sqrt(num_words)))

    for t in range(num_words):
        plt.subplot(grid_size, grid_size, t + 1)

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
    plt.tight_layout()
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

        # FIX: Move the generated sequence to the same device as the model/features
        seq_cuda = seq.to(fc_feats.device)

        # Then, use _forward to get attention weights for the generated sequence
        # In eval mode, _forward now returns (logprobs, att_weights)
        _, output_att = model._forward(fc_feats, att_feats, seq_cuda, att_masks)

    return seq, output_att


def get_attention_weights_from_sequence(model, fc_feats, att_feats, att_masks, seq):
    """
    Gets attention weights for a given sequence.
    """
    model.eval()
    with torch.no_grad():
        # Ensure seq is on the correct device before passing to _forward
        seq_cuda = seq.to(fc_feats.device)
        # Use _forward to get attention weights for a given sequence
        _, output_att = model._forward(fc_feats, att_feats, seq_cuda, att_masks)
    return output_att