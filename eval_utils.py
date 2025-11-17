# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import time
import numpy as np
import torch

import misc.utils as utils

# Try to import COCO evaluation tools if available
try:
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap
    _HAS_COCO = True
except Exception:
    _HAS_COCO = False


bad_endings = ['a', 'an', 'the', 'with', 'in', 'on', 'at', 'for', 'from', 'by', "are", "am"]
bad_endings += ['the']


def count_bad(sen):
    """Return 1 if sentence ends with a 'bad' ending token."""
    toks = sen.strip().split(' ')
    if len(toks) == 0:
        return 1
    return 1 if toks[-1] in bad_endings else 0


def language_eval(dataset, preds, model_id, split, name=''):
    """
    Evaluate the generated captions using COCO eval if available.
    dataset: path to reference coco json or dataset dict (if repo code expects different, pass appropriate)
    preds: list of dicts {'image_id': id, 'caption': caption}
    model_id: string id for naming outputs
    split: split name for filenames
    name: optional name appended to outfile
    Returns: language_stats dict (or {} if not available)
    """
    outfile_path = os.path.join('eval_results', model_id + name + split + '.json')
    os.makedirs('eval_results', exist_ok=True)
    with open(outfile_path, 'w') as outfile:
        json.dump(preds, outfile)

    if not _HAS_COCO:
        print("pycocotools/pycocoevalcap not available, skipping COCO language evaluation.")
        return {}

    # load ground truth coco json if dataset is a path
    if isinstance(dataset, str):
        coco = COCO(dataset)
    elif isinstance(dataset, dict):
        # write a temporary json file so COCO can read it (if small)
        tmp_path = os.path.join('eval_results', 'tmp_dataset_for_coco.json')
        with open(tmp_path, 'w') as f:
            json.dump(dataset, f)
        coco = COCO(tmp_path)
    else:
        raise ValueError('dataset must be a path or a dict')

    cocoRes = coco.loadRes(outfile_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = [p['image_id'] for p in preds]
    cocoEval.evaluate()

    # build return dict
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    # compute a "bad ending rate"
    if len(preds) > 0:
        out['bad_count_rate'] = sum([count_bad(_['caption']) for _ in preds]) / float(len(preds))
    else:
        out['bad_count_rate'] = 0.0

    return out


def eval_split(model, crit, loader, eval_kwargs={}, train_mode=True):
    """
    Evaluate model on a split. This function is designed to work with both:
    - DataLoader (pre-extracted features) which returns torch tensors
    - DataLoaderRaw which returns numpy arrays for fc/att features

    Returns:
      loss_avg: average loss (if computed), otherwise 0
      predictions: list of dict {'image_id': id, 'caption': caption}
      lang_stats: language evaluation metrics dict (or None)
    """
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 1)
    verbose_loss = eval_kwargs.get('verbose_loss', 0)
    dump_json = eval_kwargs.get('dump_json', 0)
    dataset = eval_kwargs.get('dataset', eval_kwargs.get('input_json', ''))
    model_id = eval_kwargs.get('id', '')
    split = eval_kwargs.get('split', 'test')
    name = eval_kwargs.get('name', '')

    model.eval()

    # number of images to use; -1 means all
    val_images_use = eval_kwargs.get('val_images_use', -1)
    if val_images_use == -1:
        val_images_use = 1e9

    # bookkeeping
    total_loss = 0.0
    loss_evals = 0
    predictions = []
    lang_stats = None

    loader.reset_iterator(split)

    def _to_tensor(x):
        if x is None:
            return None
        # if numpy array, convert
        if isinstance(x, np.ndarray):
            t = torch.from_numpy(x)
        elif torch.is_tensor(x):
            t = x
        else:
            # try to construct tensor
            try:
                t = torch.tensor(x)
            except Exception:
                # fallback: return original object
                return x
        if torch.cuda.is_available():
            t = t.cuda()
        return t

    sent_idx = 0
    wrapped = False
    start_time = time.time()

    while True:
        # stop conditions
        if sent_idx >= val_images_use:
            break
        data = loader.get_batch(split, eval_kwargs.get('batch_size', None))

        # Convert features/masks (handle both numpy arrays and torch tensors)
        # Select indices corresponding to images (some loaders store seq_per_img)
        batch_size = loader.batch_size
        seq_per_img = getattr(loader, 'seq_per_img', 1)
        index = np.arange(batch_size) * seq_per_img

        # data fields may already be tensors or numpy arrays
        fc_np = data.get('fc_feats')
        att_np = data.get('att_feats')
        masks_np = data.get('att_masks') if 'att_masks' in data else None

        # If fc/att are arrays shaped (batch, ...) we can use them directly;
        # if they are larger (e.g., seq_per_img repeats), use index selection
        try:
            # If shapes are compatible, index rows
            if isinstance(fc_np, np.ndarray) and fc_np.shape[0] >= batch_size:
                fc_batch = fc_np[index]
            elif torch.is_tensor(fc_np) and fc_np.shape[0] >= batch_size:
                fc_batch = fc_np[index]
            else:
                fc_batch = fc_np
        except Exception:
            fc_batch = fc_np

        try:
            if isinstance(att_np, np.ndarray) and att_np.shape[0] >= batch_size:
                att_batch = att_np[index]
            elif torch.is_tensor(att_np) and att_np.shape[0] >= batch_size:
                att_batch = att_np[index]
            else:
                att_batch = att_np
        except Exception:
            att_batch = att_np

        try:
            if masks_np is not None:
                if isinstance(masks_np, np.ndarray) and masks_np.shape[0] >= batch_size:
                    masks_batch = masks_np[index]
                elif torch.is_tensor(masks_np) and masks_np.shape[0] >= batch_size:
                    masks_batch = masks_np[index]
                else:
                    masks_batch = masks_np
            else:
                masks_batch = None
        except Exception:
            masks_batch = masks_np

        # Convert to tensors and move to GPU if available
        fc = _to_tensor(fc_batch)
        att = _to_tensor(att_batch)
        masks = _to_tensor(masks_batch)

        # If ground truth labels are provided and verbose_loss requested, compute loss
        if verbose_loss and 'labels' in data and 'gts' in data:
            # labels might be stored as numpy arrays; convert
            labels = _to_tensor(data.get('labels'))
            masks_labels = _to_tensor(data.get('masks')) if 'masks' in data else None
            with torch.no_grad():
                # Many models accept (fc_feats, att_feats, labels, masks)
                try:
                    out = model(fc, att, labels, masks_labels)
                    loss = crit(out, labels)
                except Exception:
                    # fallback: try to compute loss via separate interface if present
                    loss = 0
            total_loss += float(loss) if torch.is_tensor(loss) else loss
            loss_evals += 1

        # Generate samples from the model. Prefer model.sample if available.
        with torch.no_grad():
            try:
                if hasattr(model, 'sample'):
                    # pass eval kwargs directly to sample (many models accept most options via opt dict)
                    seq, seqLogprobs = model.sample(fc, att, masks, eval_kwargs)
                else:
                    # fallback: call forward and try to get 'seq' out
                    out = model(fc, att, None, None)
                    if isinstance(out, dict) and 'seq' in out:
                        seq = out['seq']
                    else:
                        # last resort: try model.sample without kwargs
                        seq, seqLogprobs = model.sample(fc, att, masks)
            except Exception as e:
                # If sample fails, raise a clear error
                raise RuntimeError("Model sampling failed: {}".format(e))

        # seq is expected to be a tensor (seq_length x batch) or (batch x seq_length)
        if torch.is_tensor(seq):
            # normalize shape to (batch, seq_length)
            if seq.dim() == 2:
                if seq.shape[0] == batch_size:
                    seq_batch = seq
                else:
                    # common shape is (seq_length, batch)
                    seq_batch = seq.transpose(0, 1)
            else:
                seq_batch = seq
        else:
            # If seq is numpy array
            if isinstance(seq, np.ndarray):
                if seq.shape[0] == batch_size:
                    seq_batch = torch.from_numpy(seq)
                else:
                    seq_batch = torch.from_numpy(seq).transpose(0, 1)
                if torch.cuda.is_available():
                    seq_batch = seq_batch.cuda()
            else:
                # Unknown seq type
                raise RuntimeError("Unknown seq type returned by model.sample: {}".format(type(seq)))

        # decode sequences and collect predictions
        # seq_batch may be on GPU; bring to CPU for decode
        seq_cpu = seq_batch.detach().cpu().numpy()
        for i in range(seq_cpu.shape[0]):
            s = seq_cpu[i]
            # decode to words using loader vocab
            # utils.decode_sequence expects ix_to_word mapping
            try:
                sentence = utils.decode_sequence(loader.get_vocab(), s)
            except Exception:
                # try loader.ix_to_word as fallback
                try:
                    sentence = utils.decode_sequence(loader.ix_to_word, s)
                except Exception:
                    sentence = " ".join([str(int(x)) for x in s])
            # sentence may be list returned by decode_sequence; ensure string
            if isinstance(sentence, list):
                sentence = sentence[0]
            image_id = data['infos'][i]['id'] if 'infos' in data and len(data['infos']) > i else None
            predictions.append({'image_id': image_id, 'caption': sentence})
            sent_idx += 1

        # optionally print beams (if model stores done_beams after sample)
        if verbose_beam and hasattr(model, 'done_beams'):
            try:
                for i in range(len(model.done_beams)):
                    beams = model.done_beams[i]
                    print("Image %s beams:" % (data['infos'][i]['file_path'] if 'infos' in data and len(data['infos']) > i else i))
                    for b in beams:
                        seq_decoded = utils.decode_sequence(loader.get_vocab(), b['seq'].unsqueeze(0))[0]
                        print("  [score %.3f] %s" % (b.get('score', 0), seq_decoded))
            except Exception:
                # non-fatal if printing fails
                pass

        # break when we've processed enough images or if loader wrapped
        if data.get('bounds', None) is not None:
            if data['bounds'].get('wrapped', False):
                wrapped = True
        if wrapped:
            break
        if sent_idx >= val_images_use:
            break

    # compute language metrics
    try:
        lang_stats = language_eval(dataset, predictions, model_id, split, name)
    except Exception:
        lang_stats = None

    loss_avg = total_loss / loss_evals if loss_evals > 0 else 0.0

    if dump_json:
        os.makedirs('vis', exist_ok=True)
        json.dump(predictions, open('vis/vis.json', 'w'))

    return loss_avg, predictions, lang_stats