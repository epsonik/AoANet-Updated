"""
Script to visualize attention weights for image captioning.
This script loads a trained model, generates captions for images,
and creates attention heatmap visualizations.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import torch
import numpy as np
from PIL import Image

import opts
import models
from dataloaderraw import DataLoaderRaw
import misc.utils as utils
import vis_utils


def main(opt):
    # Load model and info
    print(f"Loading model from: {opt.model}")
    with open(opt.infos_path, 'rb') as f:
        infos = utils.pickle_load(f)
    
    # Override and collect parameters
    replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir', 'input_label_h5', 
               'input_json', 'batch_size', 'id']
    ignore = ['start_from']
    
    for k in vars(infos['opt']).keys():
        if k in replace:
            setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))
        elif k not in ignore:
            if k not in vars(opt):
                vars(opt).update({k: vars(infos['opt'])[k]})
    
    vocab = infos['vocab']  # ix -> word mapping
    
    # Setup the model
    opt.vocab = vocab
    model = models.setup(opt)
    del opt.vocab
    model.load_state_dict(torch.load(opt.model))
    model.cuda()
    model.eval()
    
    print("Model loaded successfully")
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create data loader for the image
    if opt.image_folder:
        loader = DataLoaderRaw({
            'folder_path': opt.image_folder,
            'coco_json': opt.coco_json,
            'batch_size': 1,
            'cnn_model': opt.cnn_model
        })
        loader.ix_to_word = vocab
    else:
        raise ValueError("Must specify --image_folder")
    
    # Create output directory
    os.makedirs(opt.output_dir, exist_ok=True)
    
    # Process images
    loader.reset_iterator('test')
    num_processed = 0
    
    print(f"\nProcessing images from: {opt.image_folder}")
    print(f"Saving visualizations to: {opt.output_dir}")
    print(f"Number of images to process: {opt.num_images if opt.num_images > 0 else 'all'}\n")
    
    while True:
        # Get batch
        data = loader.get_batch('test')
        
        # Extract features
        tmp = [data['fc_feats'][0:1], data['att_feats'][0:1],
               data['att_masks'][0:1] if data['att_masks'] is not None else None]
        tmp = [_.cuda() if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, att_masks = tmp
        
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
                     'temperature': opt.temperature}
            )
        
        # Decode sequence to words
        sents = utils.decode_sequence(vocab, seq)
        caption = sents[0]
        
        print(f"Generated caption: {caption}")
        
        # If no attention was captured (e.g., for multi-headed attention during beam search),
        # try extracting attention by re-running with the sequence
        if len(attention_weights) == 0:
            print("No attention captured during generation, extracting from sequence...")
            attention_weights = vis_utils.get_attention_weights_from_sequence(
                model, fc_feats, att_feats, att_masks, seq
            )
        
        if len(attention_weights) > 0:
            # Split caption into words
            words = caption.split()
            
            # Adjust attention_weights list to match words (handle potential mismatches)
            min_len = min(len(attention_weights), len(words))
            attention_weights = attention_weights[:min_len]
            words = words[:min_len]
            
            print(f"Creating visualizations for {len(words)} words...")
            
            # Get the actual image path
            if opt.image_folder:
                actual_image_path = os.path.join(opt.image_folder, image_path)
            else:
                actual_image_path = image_path
            
            # Check if image exists
            if not os.path.exists(actual_image_path):
                print(f"Warning: Image not found at {actual_image_path}")
                # Try to find the image in the folder
                possible_path = os.path.join(opt.image_folder, os.path.basename(image_path))
                if os.path.exists(possible_path):
                    actual_image_path = possible_path
                    print(f"Found image at: {actual_image_path}")
                else:
                    print(f"Skipping visualization for this image")
                    num_processed += 1
                    continue
            
            # Determine attention size from features
            att_size = att_feats.size(1)
            
            # Create visualizations
            vis_paths = vis_utils.visualize_attention_for_sequence(
                actual_image_path,
                attention_weights,
                words,
                output_dir=opt.output_dir,
                att_size=att_size
            )
            
            print(f"Created {len(vis_paths)} visualization(s)")
        else:
            print("Warning: Could not extract attention weights for this image")
        
        num_processed += 1
        
        print()  # Empty line for readability
        
        # Check if we should stop
        if opt.num_images > 0 and num_processed >= opt.num_images:
            break
        
        if data['bounds']['wrapped']:
            break
    
    print(f"\nProcessing complete! Processed {num_processed} image(s)")
    print(f"Visualizations saved to: {opt.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Model parameters
    parser.add_argument('--model', type=str, required=True,
                        help='path to model checkpoint (.pth file)')
    parser.add_argument('--infos_path', type=str, required=True,
                        help='path to infos file (.pkl file)')
    parser.add_argument('--cnn_model', type=str, default='resnet101',
                        help='CNN model for feature extraction (resnet101, resnet152, etc.)')
    
    # Input image parameters
    parser.add_argument('--image_folder', type=str, required=True,
                        help='folder containing images to visualize')
    parser.add_argument('--coco_json', type=str, default='',
                        help='optional: coco json file for image info')
    if '--num_images' not in parser._option_string_actions:
        parser.add_argument('--num_images', type=int, default=1,
                            help='number of images to process (-1 for all)')

    # Generation parameters
    parser.add_argument('--sample_method', type=str, default='beam',
                        help='sampling method (greedy, beam, etc.)')
    parser.add_argument('--beam_size', type=int, default=3,
                        help='beam size for beam search (1 = greedy)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature for sampling')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='vis/attention',
                        help='directory to save attention visualizations')
    
    # Parse arguments
    opt = parser.parse_args()
    
    # Add any missing opts that models.setup expects
    opts.add_eval_options(parser)
    opt = parser.parse_args()
    
    # Run visualization
    main(opt)
