# Copyright (c) 2017, FAIR anaf
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os
import six.moves.urllib as urllib
import six
if six.PY2:
    import cPickle as pickle
else:
    import pickle


def pickle_load(f):
    """ Load a pickle.
    Parameters
    ----------
    f: file-like object
    """
    if six.PY2:
        return pickle.load(f)
    elif six.PY3:
        return pickle.load(f, encoding='latin-1')
    raise ValueError("invalid python version: {}".format(sys.version))


def pickle_dump(obj, f):
    """ Dump a pickle.
    Parameters
    ----------
    obj: object
    f: file-like object
    """
    if six.PY2:
        return pickle.dump(obj, f, protocol=2)
    elif six.PY3:
        return pickle.dump(obj, f, protocol=2)
    raise ValueError("invalid python version: {}".format(sys.version))


bad_endings = ['a','an','the','in','for','at','of','with','before','after','on','upon','near','to','is','are','am']
bad_endings += ['the']

def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j].item()
            if ix > 0:
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix)]
            else:
                # ix is 0 or less, assume it's the EOS token
                if j >= 1:
                    txt = txt + ' '
                txt = txt + '<eos>'
                break  # Stop after adding EOS
        if int(os.getenv('REMOVE_BAD_ENDINGS', '0')):
            flag = 0
            words = txt.split(' ')
            for j in range(len(words)):
                if words[-j-1] not in bad_endings:
                    flag = -j
                    break
            txt = ' '.join(words[0:len(words)+flag])
        out.append(txt.replace('@@ ', ''))
    return out

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def get_lr(optimizer):
    for group in optimizer.param_groups:
        return group['lr']

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def L2_pen(models):
    L2_p = 0
    for model in models:
        for name, param in model.named_parameters():
            if 'bias' not in name:
                L2_p += (param ** 2).sum()
    return L2_p

def save_checkpoint(model, infos, optimizer, histories=None, append=''):
    if len(append) > 0:
        append = '-' + append
    # if checkpoint_path doesn't exist
    if not os.path.isdir(infos['opt'].checkpoint_path):
        os.makedirs(infos['opt'].checkpoint_path)
    checkpoint_path = os.path.join(infos['opt'].checkpoint_path, 'model%s.pth' %(append))
    torch.save(model.state_dict(), checkpoint_path)
    print("model saved to {}".format(checkpoint_path))

    #
    if histories:
        histories_path = os.path.join(infos['opt'].checkpoint_path, 'histories%s.pkl' %(append))
        with open(histories_path, 'wb') as f:
            pickle_dump(histories, f)

    # with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'.pkl'), 'wb') as f:
    with open(os.path.join(infos['opt'].checkpoint_path, 'infos%s.pkl' %(append)), 'wb') as f:
        pickle_dump(infos, f)

def LanguageModelCriterion():
    return nn.CrossEntropyLoss()
