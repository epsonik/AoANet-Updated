"""
Script to visualize attention weights for image captioning.
This script loads a trained model, generates captions for images,
creates attention heatmap visualizations, and (optionally)
computes per-image captioning metrics (BLEU, METEOR, ROUGE, CIDEr)
using COCO annotations, saving them to CSV files.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil
import torch
import numpy as np
from PIL import Image
import json
import csv
import tempfile

import opts
import models
from dataloaderraw import DataLoaderRaw
import misc.utils as utils
import vis_utils

# Optional imports for COCO evaluation
try:
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap
    COCO_EVAL_AVAILABLE = True
except Exception:
    COCO_EVAL_AVAILABLE = False


def save_metrics_csv(metrics_dict, csv_path):
    """
    Save metrics (dict) to csv file with two columns: Metric, Score.
    """
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Score'])
        for k, v in metrics_dict.items():
            writer.writerow([k, v])


def compute_and_save_coco_metrics(coco, image_id, pred_caption, output_dir):
    """
    Compute COCO captioning metrics for a single image using pycocoevalcap.
    Saves per-image metrics to CSV in output_dir and returns the dict of metrics.
    """
    # Prepare predictions as a list as expected by COCO API
    preds = [{'image_id': int(image_id), 'caption': pred_caption}]

    # Write to a temporary json file for coco.loadRes
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmpf:
        json.dump(preds, tmpf)
        tmpf_path = tmpf.name

    try:
        coco_res = coco.loadRes(tmpf_path)
        coco_eval = COCOEvalCap(coco, coco_res)
        # restrict to our single image id
        coco_eval.params['image_id'] = [int(image_id)]
        coco_eval.evaluate()

        # coco_eval.imgToEval contains per-image metrics
        img_to_eval = coco_eval.imgToEval
        metrics = {}
        if int(image_id) in img_to_eval:
            metrics = img_to_eval[int(image_id)]
        else:
            # sometimes keys are strings
            metrics = img_to_eval.get(str(image_id), {})

        # Save CSV
        csv_path = os.path.join(output_dir, f"{image_id}_metrics.csv")
        save_metrics_csv(metrics, csv_path)
        return metrics
    finally:
        try:
            os.remove(tmpf_path)
        except Exception:
            pass


def main(opt):
    # load infos
    infos = utils.load_info(opt.infos_path)
    vocab = infos.get('vocab', None)
    if vocab is None:
        # legacy: some versions store 'ix_to_word' or 'vocab' differently
        vocab = infos.get('ix_to_word', None)
    if isinstance(vocab, dict):
        # utils.decode_sequence expects ix_to_word mapping (index->word)
        ix_to_word = vocab
    else:
        ix_to_word = infos.get('vocab', None)

    # build model
    model = models.setup(opt)
    model.eos_idx = getattr(infos['opt'], 'eos_idx', 0)
    model.pad_idx = getattr(infos['opt'], 'pad_idx', 0)

    model.load_state_dict(torch.load(opt.model, map_location=torch.device('cpu')))
    model.cuda()
    model.eval()

    print("Model loaded successfully")
    if ix_to_word is not None:
        try:
            print(f"Vocabulary size: {len(ix_to_word)}")
        except Exception:
            pass

    # Create data loader for the image
    if opt.image_folder:
        loader = DataLoaderRaw({
            'folder_path': opt.image_folder,
            'coco_json': opt.coco_json,
            'batch_size': 1,
            'cnn_model': opt.cnn_model
        })
        # ensure loader has vocab mapping if present in infos
        if ix_to_word is not None:
            loader.ix_to_word = ix_to_word
    else:
        raise ValueError("Must specify --image_folder")

    # Create output directory
    os.makedirs(opt.output_dir, exist_ok=True)

    # Optionally load COCO annotations if provided and pycocotools available
    coco = None
    if opt.coco_json:
        if not COCO_EVAL_AVAILABLE:
            print("Warning: pycocotools / pycocoevalcap not available. Per-image metrics will be skipped.")
        else:
            print(f"Loading COCO annotations from: {opt.coco_json}")
            coco = COCO(opt.coco_json)

    # Process images
    loader.reset_iterator('test')
    num_processed = 0

    print(f"\nProcessing images from: {opt.image_folder}")
    while True:
        # Get batch
        data = loader.get_batch('test')

        def to_cuda_tensor(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x.cuda()
            try:
                t = torch.from_numpy(x)
                return t.cuda()
            except Exception:
                return x

        fc_feats = to_cuda_tensor(data['fc_feats'][0:1])
        att_feats = to_cuda_tensor(data['att_feats'][0:1])
        att_masks = to_cuda_tensor(data.get('att_masks')[0:1]) if data.get('att_masks') is not None else None

        # Get image info
        image_id = data['infos'][0]['id']
        image_path = data['infos'][0]['file_path']

        print(f"Processing image {num_processed + 1}: {image_path}")

        # Generate caption and capture attention
        with torch.no_grad():
            seq, attention_weights = vis_utils.capture_attention_weights(
                model, fc_feats, att_feats, att_masks,
                opt={'sample_method': opt.sample_method,
                     'beam_size': opt.beam_size,
                     'temperature': opt.temperature,
                     'sample_n': opt.sample_n,
                     'sample_max': opt.sample_max}
            )

        # Decode the sequence to string(s)
        try:
            sents = utils.decode_sequence(loader.ix_to_word if hasattr(loader, 'ix_to_word') else ix_to_word, seq)
        except Exception:
            # fallback: try using infos vocab
            sents = utils.decode_sequence(ix_to_word, seq)
        caption = sents[0] if isinstance(sents, (list, tuple)) and len(sents) > 0 else str(sents)

        # Get the actual image path (the dataloader returns the full path when image_folder is used)
        if opt.image_folder:
            actual_image_path = image_path
        else:
            actual_image_path = image_path

        # Check if image exists
        if not os.path.exists(actual_image_path):
            print(f"Warning: Image not found at {actual_image_path}")
            possible_path = os.path.join(opt.image_folder, os.path.basename(image_path))
            if os.path.exists(possible_path):
                actual_image_path = possible_path
                print(f"Found image at: {actual_image_path}")
            else:
                print(f"Skipping visualization for this image")
                num_processed += 1
                # Check limits and continue
                if opt.num_images > 0 and num_processed >= opt.num_images:
                    break
                if data['bounds']['wrapped']:
                    break
                continue

        # Determine attention size from features
        att_size = att_feats.size(1)

        # Create a subdirectory for this image based on its filename (without extension) or coco id
        # If filenames are COCO ids as requested, use the id; otherwise use basename
        try:
            image_basename = os.path.splitext(os.path.basename(actual_image_path))[0]
        except Exception:
            image_basename = str(image_id)
        image_output_dir = os.path.join(opt.output_dir, image_basename)
        os.makedirs(image_output_dir, exist_ok=True)

        # Copy the original image to the subdirectory
        original_image_dest = os.path.join(image_output_dir, 'original.jpg')
        try:
            shutil.copy2(actual_image_path, original_image_dest)
        except Exception:
            # if copy fails, ignore
            pass

        # If COCO annotations are available, compute per-image metrics and save CSV
        if coco is not None:
            try:
                metrics = compute_and_save_coco_metrics(coco, image_id, caption, image_output_dir)
                print(f"Saved metrics CSV for image {image_id} to {image_output_dir}")
            except Exception as e:
                print(f"Warning: Failed to compute/save COCO metrics for image {image_id}: {e}")

        # Create visualizations (attention heatmaps + per-token images)
        vis_paths = vis_utils.visualize_attention_for_sequence(
            actual_image_path,
            attention_weights,
            caption.split(),  # pass the words sequence so vis_utils labels per-step images
            output_dir=image_output_dir,
            att_size=att_size
        )

        print(f"Created {len(vis_paths)} visualization(s) in {image_output_dir}")

        num_processed += 1

        print()  # Empty line for readability

        # Check if we should stop
        if opt.num_images > 0 and num_processed >= opt.num_images:
            break

        if data['bounds']['wrapped']:
            break

    print(f"\nProcessing complete! Processed {num_processed} image(s)")
    print(f"Visualizations and metrics (if any) saved to: {opt.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument('--model', type=str, required=True,
                        help='path to model checkpoint (.pth file)')
    parser.add_argument('--infos_path', type=str, required=True,
                        help='path to infos file (.pkl file)')
    parser.add_argument('--cnn_model', type=str, default='densenet201',
                        help='CNN model for feature extraction (resnet101, resnet152, etc.)')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='vis/attention',
                        help='directory to save attention visualizations')

    opts.add_eval_options(parser)

    opt = parser.parse_args()

    # Run visualization
    main(opt)