# python
"""
Script to visualize attention weights for image captioning.
Creates a subdirectory per input image (basename) under `opt.output_dir`
and copies the original image into that folder named `original<ext>`.
Saves per-word attention visualizations into the same per-image folder.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil
import torch
from PIL import Image

import opts
import models
from dataloaderraw import DataLoaderRaw
import misc.utils as utils
import vis_utils


def to_cuda_tensor(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.cuda()
    return torch.from_numpy(x).float().cuda()


def main(opt):
    # Load model and infos
    print(f"Loading model from: {opt.model}")
    with open(opt.infos_path, 'rb') as f:
        infos = utils.pickle_load(f)

    # Merge/override options
    replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir', 'input_label_h5',
               'input_json', 'batch_size', 'id']
    ignore = ['start_from']
    for k in vars(infos['opt']).keys():
        if k in replace:
            setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))
        elif k not in ignore:
            if k not in vars(opt):
                vars(opt).update({k: vars(infos['opt'])[k]})

    vocab = infos['vocab']

    # Setup model
    opt.vocab = vocab
    model = models.setup(opt)
    del opt.vocab

    model.bos_idx = getattr(infos['opt'], 'bos_idx', 0)
    model.eos_idx = getattr(infos['opt'], 'eos_idx', 0)
    model.pad_idx = getattr(infos['opt'], 'pad_idx', 0)

    model.load_state_dict(torch.load(opt.model, map_location=torch.device('cpu')))
    model.cuda()
    model.eval()

    print("Model loaded successfully")
    print(f"Vocabulary size: {len(vocab)}")

    # Data loader
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

    os.makedirs(opt.output_dir, exist_ok=True)

    loader.reset_iterator('test')
    num_processed = 0

    print(f"\nProcessing images from: {opt.image_folder}")
    print(f"Saving visualizations to: {opt.output_dir}")
    print(f"Number of images to process: {opt.num_images if opt.num_images > 0 else 'all'}\n")

    while True:
        data = loader.get_batch('test')

        fc_feats = to_cuda_tensor(data['fc_feats'][0:1])
        att_feats = to_cuda_tensor(data['att_feats'][0:1])
        att_masks = to_cuda_tensor(data.get('att_masks')[0:1]) if data.get('att_masks') is not None else None

        image_id = data['infos'][0]['id']
        image_path = data['infos'][0]['file_path']

        print(f"Processing image {num_processed + 1}: {image_path}")

        # Generate caption and try to capture attention
        with torch.no_grad():
            seq, attention_weights = vis_utils.capture_attention_weights(
                model, fc_feats, att_feats, att_masks,
                opt={'sample_method': opt.sample_method,
                     'beam_size': opt.beam_size,
                     'temperature': opt.temperature}
            )

        sents = utils.decode_sequence(vocab, seq)
        caption = sents[0]
        print(f"Generated caption: {caption}")

        if len(attention_weights) == 0:
            print("No attention captured during generation, extracting from sequence...")
            attention_weights = vis_utils.get_attention_weights_from_sequence(
                model, fc_feats, att_feats, att_masks, seq
            )

        if len(attention_weights) > 0:
            words = caption.split()
            min_len = min(len(attention_weights), len(words))
            attention_weights = attention_weights[:min_len]
            words = words[:min_len]

            print(f"Creating visualizations for {len(words)} words...")

            actual_image_path = image_path if opt.image_folder else image_path

            # Ensure the image path exists; try fallback in image_folder
            if not os.path.exists(actual_image_path):
                print(f"Warning: Image not found at {actual_image_path}")
                possible_path = os.path.join(opt.image_folder, os.path.basename(image_path))
                if os.path.exists(possible_path):
                    actual_image_path = possible_path
                    print(f"Found image at: {actual_image_path}")
                else:
                    print("Skipping visualization for this image")
                    num_processed += 1
                    continue

            # Per-image output directory named after the original image basename (no ext)
            image_basename = os.path.splitext(os.path.basename(actual_image_path))[0]
            per_image_dir = os.path.join(opt.output_dir, image_basename)
            os.makedirs(per_image_dir, exist_ok=True)

            # Copy original image into per-image folder and name it "original<ext>"
            try:
                _, ext = os.path.splitext(actual_image_path)
                dest_path = os.path.join(per_image_dir, f'original{ext}')
                shutil.copy2(actual_image_path, dest_path)
            except Exception as e:
                print(f"Warning: could not copy original image to {per_image_dir}: {e}")

            # Determine attention grid size (number of spatial locations)
            # att_feats shape expected: (1, num_loc, feat_dim)
            att_size = att_feats.size(1) if att_feats is not None else None

            vis_paths = vis_utils.visualize_attention_for_sequence(
                actual_image_path,
                attention_weights,
                words,
                output_dir=per_image_dir,
                att_size=att_size
            )

            print(f"Created {len(vis_paths)} visualization(s) in {per_image_dir}")
        else:
            print("Warning: Could not extract attention weights for this image")

        num_processed += 1
        print()

        if opt.num_images > 0 and num_processed >= opt.num_images:
            break

        if data['bounds']['wrapped']:
            break

    print(f"\nProcessing complete! Processed {num_processed} image(s)")
    print(f"Visualizations saved to: {opt.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=True,
                        help='path to model checkpoint (.pth file)')
    parser.add_argument('--infos_path', type=str, required=True,
                        help='path to infos file (.pkl file)')
    parser.add_argument('--cnn_model', type=str, default='densenet201',
                        help='CNN model for feature extraction (resnet101, resnet152, etc.)')

    parser.add_argument('--output_dir', type=str, default='vis/attention',
                        help='directory to save attention visualizations')

    opts.add_eval_options(parser)

    opt = parser.parse_args()
    main(opt)
