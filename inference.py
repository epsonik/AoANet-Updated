from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import re
import sys

import numpy as np
import torch

import models
import misc.utils as utils
from dataloaderraw import DataLoaderRaw


def extract_image_id(filename):
    """
    Extract COCO image ID from filename.
    Examples: 
        COCO_val2014_000000391895.jpg -> 391895
        COCO_val2014_000000123456.jpg -> 123456
    """
    match = re.search(r'COCO_val2014_(\d+)', filename)
    if match:
        return int(match.group(1))
    return None


def load_reference_captions(caption_file):
    """
    Load reference captions from COCO annotation file.
    Returns a dictionary mapping image_id to list of captions.
    """
    print(f"Loading reference captions from {caption_file}...")
    with open(caption_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create mapping from image_id to captions
    image_to_captions = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        caption = ann['caption']
        if image_id not in image_to_captions:
            image_to_captions[image_id] = []
        image_to_captions[image_id].append(caption)
    
    print(f"Loaded {len(image_to_captions)} images with reference captions")
    return image_to_captions


def evaluate_captions(predictions, reference_captions_file):
    """
    Evaluate generated captions against reference captions using COCO metrics.
    
    Args:
        predictions: List of dicts with 'image_id' and 'caption' keys
        reference_captions_file: Path to captions_val2014.json
    
    Returns:
        Dictionary with evaluation scores
    """
    sys.path.append("coco-caption")
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap
    
    # Load COCO annotations
    coco = COCO(reference_captions_file)
    
    # Filter predictions to only those with reference captions
    valid_image_ids = set(coco.getImgIds())
    predictions_filt = [p for p in predictions if p['image_id'] in valid_image_ids]
    
    print(f"\nEvaluating {len(predictions_filt)}/{len(predictions)} predictions with reference captions")
    
    if len(predictions_filt) == 0:
        print("Warning: No predictions match reference captions!")
        return None
    
    # Save predictions to temporary file
    temp_file = '/tmp/predictions_temp.json'
    with open(temp_file, 'w') as f:
        json.dump(predictions_filt, f)
    
    # Load predictions and evaluate
    coco_result = coco.loadRes(temp_file)
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.params['image_id'] = coco_result.getImgIds()
    coco_eval.evaluate()
    
    # Clean up temporary file
    os.remove(temp_file)
    
    return coco_eval.eval


def main(args):
    # Load model info
    print(f"Loading model info from {args.infos_path}...")
    with open(args.infos_path, 'rb') as f:
        infos = utils.pickle_load(f)
    
    vocab = infos['vocab']
    
    # Setup model
    opt = infos['opt']
    opt.vocab = vocab
    
    print(f"Loading model from {args.model}...")
    model = models.setup(opt)
    model.load_state_dict(torch.load(args.model))
    model.cuda()
    model.eval()
    
    print(f"Model loaded successfully")
    
    # Setup data loader for raw images
    loader_opt = {
        'folder_path': args.image_folder,
        'coco_json': args.coco_json if hasattr(args, 'coco_json') and args.coco_json else '',
        'batch_size': args.batch_size,
        'cnn_model': args.cnn_model
    }
    
    print(f"Loading images from {args.image_folder}...")
    loader = DataLoaderRaw(loader_opt)
    loader.ix_to_word = vocab
    
    # Generate captions
    print(f"\nGenerating captions...")
    predictions = []
    loader.reset_iterator('val')
    
    while True:
        data = loader.get_batch('val')
        
        # Get features
        fc_feats = torch.from_numpy(data['fc_feats']).cuda()
        att_feats = torch.from_numpy(data['att_feats']).cuda()
        att_masks = data['att_masks']
        if att_masks is not None:
            att_masks = torch.from_numpy(att_masks).cuda()
        
        # Generate captions
        with torch.no_grad():
            seq, _ = model(fc_feats, att_feats, att_masks, 
                          opt={'beam_size': args.beam_size, 'sample_method': 'beam_search'},
                          mode='sample')
        
        # Decode sequences
        sents = utils.decode_sequence(vocab, seq)
        
        # Process results
        for k, sent in enumerate(sents):
            info = data['infos'][k]
            image_id = info['id']
            file_path = info['file_path']
            
            # Try to extract COCO image ID from filename if not already an integer
            if isinstance(image_id, str):
                extracted_id = extract_image_id(os.path.basename(file_path))
                if extracted_id is not None:
                    image_id = extracted_id
                else:
                    # Try to convert string to int
                    try:
                        image_id = int(image_id)
                    except ValueError:
                        print(f"Warning: Could not extract image ID from {file_path}")
                        continue
            
            entry = {
                'image_id': int(image_id),
                'caption': sent,
                'file_path': file_path
            }
            predictions.append(entry)
            
            if args.verbose:
                print(f"Image {image_id} ({os.path.basename(file_path)}): {sent}")
        
        # Check if we're done
        if data['bounds']['wrapped']:
            break
    
    print(f"\nGenerated {len(predictions)} captions")
    
    # Save predictions if requested
    if args.output_json:
        print(f"Saving predictions to {args.output_json}...")
        with open(args.output_json, 'w') as f:
            json.dump(predictions, f, indent=2)
    
    # Evaluate captions if reference file is provided
    if args.reference_captions:
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print(f"{'='*60}")
        
        scores = evaluate_captions(predictions, args.reference_captions)
        
        if scores:
            print(f"\nMetrics:")
            print(f"  BLEU-1: {scores['Bleu_1']:.4f}")
            print(f"  BLEU-2: {scores['Bleu_2']:.4f}")
            print(f"  BLEU-3: {scores['Bleu_3']:.4f}")
            print(f"  BLEU-4: {scores['Bleu_4']:.4f}")
            print(f"  METEOR: {scores['METEOR']:.4f}")
            print(f"  ROUGE_L: {scores['ROUGE_L']:.4f}")
            print(f"  CIDEr: {scores['CIDEr']:.4f}")
            if 'SPICE' in scores:
                print(f"  SPICE: {scores['SPICE']:.4f}")
            
            print(f"\n{'='*60}")
    else:
        print("\nSkipping evaluation (no reference captions provided)")
    
    print("\nInference complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate and evaluate image captions')
    
    # Model parameters
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--infos_path', type=str, required=True,
                       help='Path to model infos pickle file')
    parser.add_argument('--cnn_model', type=str, default='densenet161',
                       help='CNN model to use for feature extraction (densenet161, resnet101, etc.)')
    
    # Input parameters
    parser.add_argument('--image_folder', type=str, required=True,
                       help='Path to folder containing input images')
    parser.add_argument('--coco_json', type=str, default='',
                       help='Optional: Path to COCO-style JSON file with image info')
    parser.add_argument('--reference_captions', type=str, default='captions_val2014.json',
                       help='Path to reference captions file for evaluation (default: captions_val2014.json)')
    
    # Generation parameters
    parser.add_argument('--beam_size', type=int, default=2,
                       help='Beam size for caption generation')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for processing images')
    
    # Output parameters
    parser.add_argument('--output_json', type=str, default='',
                       help='Path to save predictions as JSON')
    parser.add_argument('--verbose', type=int, default=1,
                       help='Print captions as they are generated')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.infos_path):
        print(f"Error: Infos file not found: {args.infos_path}")
        sys.exit(1)
    
    if not os.path.exists(args.image_folder):
        print(f"Error: Image folder not found: {args.image_folder}")
        sys.exit(1)
    
    if args.reference_captions and not os.path.exists(args.reference_captions):
        print(f"Error: Reference captions file not found: {args.reference_captions}")
        sys.exit(1)
    
    main(args)
