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
    # Zmiana: Znormalizuj rozmiar obrazu do 256x256
    image = image.resize([256, 256], Image.LANCZOS)
    
    words = [word.replace('/', '') for word in words]
    
    vis_paths = []

    for t in range(len(words)):
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)
        
        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=8)
        plt.imshow(image)
        
        # Pobierz wagę atencji i przeskaluj ją
        current_alpha = attention_weights[t].cpu().numpy()
        alpha = skimage.transform.resize(current_alpha, [image.size[1], image.size[0]])
        
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.7)
        
        plt.set_cmap(plt.cm.Greys_r)
        plt.axis('off')

    # Zapisz całą sekwencję w jednym pliku
    output_path = os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_att.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    vis_paths.append(output_path)
    
    return vis_paths

# Pozostałe funkcje w pliku pozostają bez zmian
def capture_attention_weights(model, fc_feats, att_feats, att_masks, opt):
    model.eval()
    with torch.no_grad():
        seq, _, output_att = model._sample(fc_feats, att_feats, att_masks, opt)
    return seq, output_att

def get_attention_weights_from_sequence(model, fc_feats, att_feats, att_masks, seq):
    model.eval()
    with torch.no_grad():
        # Użyj _forward do uzyskania wag atencji dla danej sekwencji
        _, _, output_att = model._forward(fc_feats, att_feats, seq, att_masks)
    return output_att