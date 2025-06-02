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
from prettytable import PrettyTable


def count_parameters(model, model_name):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

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

count_parameters(model, opt['model'])
