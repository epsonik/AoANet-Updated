import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import re
import csv
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider


def resize_attention_to_image(attention, image_shape, grid_size):
    """
    Resizes the attention map from a grid to the full image size.
    """
    # The attention map is (grid_size*grid_size), reshape to (grid_size, grid_size)
    attention_map = attention.reshape(grid_size, grid_size)

    # Resize to the image dimensions
    resized_attention = cv2.resize(attention_map, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_LINEAR)

    return resized_attention


def create_heatmap(attention_map, image):
    """
    Applies a heatmap to the image based on the attention map.
    """
    # Normalize the attention map for visualization
    heatmap = (attention_map - np.min(attention_map)) / (np.max(attention_map) - np.min(attention_map))
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
        try:  # Use textbbox for more accurate size
            bbox = draw.textbbox((x, y), w, font=font)
        except AttributeError:  # Fallback for older Pillow
            text_size = draw.textsize(w, font=font)
            bbox = (x, y, x + text_size[0], y + text_size[1])

        if is_highlighted:
            draw.rectangle(bbox, fill="yellow")
            draw.text((x, y), w, font=font, fill="black")
        else:
            draw.text((x, y), w, font=font, fill="white")

        x = bbox[2] + 5  # Move x for the next word

    return image


def calculate_metrics_and_save(res, gts, image_id, output_dir):
    """
    Calculates image captioning metrics and saves them to a CSV file.
    :param res: dict with predicted caption, e.g., {<image_id>: [{'caption': <pred_caption_string>}]}
    :param gts: dict with ground truth captions, e.g., {<image_id>: [{'caption': <gt_caption_string_1>}, ...]}
    :param image_id: The ID of the image.
    :param output_dir: Directory to save the CSV file.
    """
    # Tokenize
    tokenizer = PTBTokenizer()
    res_tokenized = tokenizer.tokenize(res)
    gts_tokenized = tokenizer.tokenize(gts)

    scorers = {
        "Bleu": Bleu(4),
        "Meteor": Meteor(),
        "Rouge": Rouge(),
        "Cider": Cider()
    }

    metrics = {}
    for name, scorer in scorers.items():
        score, _ = scorer.compute_score(gts_tokenized, res_tokenized)
        if isinstance(score, list):
            for i, s in enumerate(score):
                metrics[f"{name}-{i + 1}"] = s
        else:
            metrics[name] = score

    # Save to CSV
    csv_path = os.path.join(output_dir, f"{image_id}_metrics.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Metric', 'Score'])
        for metric, score in metrics.items():
            writer.writerow([metric, score])
    print(f"Metrics for image {image_id} saved to {csv_path}")


def visualize_attention_for_sequence(image_path, attention_weights, words, output_dir, att_size, coco_annotations=None):
    """
    Creates and saves a sequence of visualizations for each word in the caption.
    Optionally calculates and saves captioning metrics if coco_annotations are provided.
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

    # --- POCZĄTEK ZMIANY: Obliczanie metryk ---
    if coco_annotations:
        # Wyodrębnij ID obrazu COCO z nazwy pliku
        match = re.search(r'COCO_.*?_(\d+)\.jpg', os.path.basename(image_path))
        if match:
            image_id = int(match.group(1))

            predicted_caption = ' '.join(words)

            # Przygotuj dane dla ewaluatora
            gts = coco_annotations.imgToAnns[image_id]
            gts_captions = {image_id: [{'caption': ann['caption']} for ann in gts]}
            res_captions = {image_id: [{'caption': predicted_caption}]}

            calculate_metrics_and_save(res_captions, gts_captions, image_id, output_dir)
    # --- KONIEC ZMIANY ---

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
        vis_path = os.path.join(output_dir, f'step_{i}_{sanitized_word}.png')
        vis_image.save(vis_path)
        vis_paths.append(vis_path)

    return vis_paths