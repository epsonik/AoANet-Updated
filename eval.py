from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import opts
import models
from dataloader import *
from dataloaderraw import *
import eval_utils
import argparse
import misc.utils as utils
import torch
import json

# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--model', type=str, default='',
                    help='path to model to evaluate')
parser.add_argument('--cnn_model', type=str, default='densenet161',
                    help='resnet101, resnet152')
parser.add_argument('--infos_path', type=str, default='',
                    help='path to infos to evaluate')
parser.add_argument('--name', type=str, default='',
                    help='')
# New args: evaluate a single (or several) image(s) by COCO id
parser.add_argument('--image_id', type=int, default=None,
                    help='ID of a single COCO image to evaluate')
parser.add_argument('--image_ids', type=str, default='',
                    help='Comma-separated list of COCO image ids to evaluate (e.g. "12345,67890")')

opts.add_eval_options(parser)

opt = parser.parse_args()

# Load infos
with open(opt.infos_path, 'rb') as f:
    infos = utils.pickle_load(f)

# override and collect parameters
replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir', 'input_label_h5', 'input_json', 'batch_size', 'id']
ignore = ['start_from']

for k in vars(infos['opt']).keys():
    if k in replace:
        setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))
    elif k not in ignore:
        if not k in vars(opt):
            vars(opt).update({k: vars(infos['opt'])[k]})  # copy over options from model
vocab = infos['vocab']  # ix -> word mapping

# Setup the model
opt.vocab = vocab
model = models.setup(opt)
del opt.vocab
model.load_state_dict(torch.load(opt.model))
model.cuda()
model.eval()
crit = utils.LanguageModelCriterion()

# Create the Data Loader instance
if len(opt.image_folder) == 0:
    loader = DataLoader(opt)
else:
    loader = DataLoaderRaw({'folder_path': opt.image_folder,
                            'coco_json': opt.coco_json,
                            'batch_size': opt.batch_size,
                            'cnn_model': opt.cnn_model})
# When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
# So make sure to use the vocab in infos file.
loader.ix_to_word = infos['vocab']

# If user asked to evaluate a specific image or images by COCO id, restrict the loader to them
if (opt.image_id is not None) or (hasattr(opt, 'image_ids') and opt.image_ids):
    ids = []
    if opt.image_id is not None:
        ids.append(opt.image_id)
    if getattr(opt, 'image_ids', None):
        # parse comma-separated ids
        ids += [int(x) for x in opt.image_ids.split(',') if x.strip()]

    # map coco image id -> index in loader.info['images']
    id2ix = {img['id']: i for i, img in enumerate(loader.info['images'])}
    wanted_indices = [id2ix[i] for i in ids if i in id2ix]

    if len(wanted_indices) == 0:
        print('No matching images found for provided id(s):', ids)
        print('Make sure the ids exist in', opt.input_json, 'and match the keys used in your feature files.')
        exit(1)

    # restrict splits to only those indices (we'll use the 'test' split)
    loader.split_ix = {'train': [], 'val': [], 'test': wanted_indices}
    loader.iterators = {'train': 0, 'val': 0, 'test': 0}

    # Recreate prefetchers for new split_ix (BlobFetcher is available via dataloader import)
    loader._prefetch_process = {}
    for split in loader.iterators.keys():
        loader._prefetch_process[split] = BlobFetcher(split, loader, split == 'train')

    # Force evaluation split and batch size appropriate for single-image eval
    opt.split = 'test'
    opt.batch_size = 1
    print('Restricted evaluation to image ids:', ids, '-> indices:', wanted_indices)

# Set sample options
opt.datset = opt.input_json
print(opt.name)
loss, split_predictions, lang_stats = eval_utils.eval_split(model, crit, loader,
                                                            vars(opt))

print('loss: ', loss)
if lang_stats:
    print(lang_stats)

if opt.dump_json == 1:
    # dump the json
    json.dump(split_predictions, open('vis/vis.json', 'w'))