# python
# File: `dataloaderraw.py`
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import sys

import numpy as np
import torch
import skimage
import skimage.io

from torchvision import transforms as trn
preprocess = trn.Compose([
        #trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class DataLoaderRaw():
    
    def __init__(self, opt):
        self.opt = opt
        self.coco_json = opt.get('coco_json', '')
        self.folder_path = opt.get('folder_path', '')

        self.batch_size = opt.get('batch_size', 1)
        self.seq_per_img = 1

        print("start")
        sys.path.append("./misc")
        from densenet161 import DenseNet161
        from densenet_utils import myDensenet
        net = DenseNet161()
        self.my_densenet = myDensenet(net)
        self.my_densenet.cuda()
        self.my_densenet.eval()



        # load the json file which contains additional information about the dataset
        print('DataLoaderRaw loading images from folder: ', self.folder_path)

        self.files = []
        self.ids = []

        print(len(self.coco_json))
        if len(self.coco_json) > 0:
            print('reading from ' + opt.coco_json)
            # read in filenames from the coco-style json file
            self.coco_annotation = json.load(open(self.coco_json))
            for k,v in enumerate(self.coco_annotation['images']):
                fullpath = os.path.join(self.folder_path, v['file_name'])
                self.files.append(fullpath)
                self.ids.append(v['id'])
        else:
            # read in all the filenames from the folder
            print('listing all images in directory ' + self.folder_path)
            def isImage(f):
                supportedExt = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.ppm','.PPM']
                for ext in supportedExt:
                    start_idx = f.rfind(ext)
                    if start_idx >= 0 and start_idx + len(ext) == len(f):
                        return True
                return False

            n = 1
            for root, dirs, files in os.walk(self.folder_path, topdown=False):
                for file in files:
                    fullpath = os.path.join(self.folder_path, file)
                    if isImage(fullpath):
                        self.files.append(fullpath)
                        self.ids.append(str(n)) # just order them sequentially
                        n = n + 1

        self.N = len(self.files)
        print('DataLoaderRaw found ', self.N, ' images')

        self.iterator = 0

    def get_batch(self, split, batch_size=None):
        batch_size = batch_size or self.batch_size

        # Do not pre-allocate fixed-size arrays; allocate on first seen feature
        fc_batch = None              # shape: (batch_size, feat_dim)
        att_batch = None             # shape: (batch_size, H, W, feat_dim)
        max_index = self.N
        wrapped = False
        infos = []

        for i in range(batch_size):
            ri = self.iterator
            ri_next = ri + 1
            if ri_next >= max_index:
                ri_next = 0
                wrapped = True
            self.iterator = ri_next

            img = skimage.io.imread(self.files[ri])

            if len(img.shape) == 2:
                img = img[:,:,np.newaxis]
                img = np.concatenate((img, img, img), axis=2)

            img = img[:,:,:3].astype('float32')/255.0
            img = torch.from_numpy(img.transpose([2,0,1])).cuda()
            img = preprocess(img)
            with torch.no_grad():
                tmp_fc, tmp_att = self.my_densenet(img)

            # FC feature -> 1D numpy
            fc_np = tmp_fc.data.cpu().float().numpy().reshape(-1)
            feat_dim = fc_np.size

            # allocate or resize fc_batch as needed
            if fc_batch is None:
                fc_batch = np.zeros((batch_size, feat_dim), dtype=np.float32)
            if feat_dim != fc_batch.shape[1]:
                new_fc = np.zeros((batch_size, feat_dim), dtype=np.float32)
                if i > 0:
                    copy_cols = min(fc_batch.shape[1], feat_dim)
                    new_fc[:i, :copy_cols] = fc_batch[:i, :copy_cols]
                fc_batch = new_fc
            fc_batch[i] = fc_np

            # Attention feature -> normalize to (H, W, C) where C == feat_dim
            att_np = tmp_att.data.cpu().float().numpy()
            # remove leading batch dim if present
            if att_np.ndim == 4 and att_np.shape[0] == 1:
                att_np = att_np[0]
            # if (C, H, W) -> transpose to (H, W, C)
            if att_np.ndim == 3:
                if att_np.shape[0] == feat_dim:
                    att_np = att_np.transpose(1, 2, 0)
                elif att_np.shape[2] == feat_dim:
                    pass  # already (H, W, C)
                else:
                    # attempt a best-effort: if first dim small (e.g. 14) treat as H
                    if att_np.shape[0] in (7, 14, 28):
                        if att_np.shape[2] == feat_dim:
                            pass
                        elif att_np.shape[1] == feat_dim:
                            att_np = att_np.transpose(0, 2, 1)
                        else:
                            att_np = att_np.transpose(1, 2, 0)
                    else:
                        att_np = att_np.transpose(1, 2, 0)
            else:
                # unexpected shape; try to flatten to a single spatial map with channels = feat_dim
                att_np = np.reshape(att_np, (1, 1, -1))

            h, w, c = att_np.shape

            # allocate or resize att_batch as needed
            if att_batch is None:
                att_batch = np.zeros((batch_size, h, w, c), dtype=np.float32)
            if (h, w, c) != tuple(att_batch.shape[1:]):
                new_att = np.zeros((batch_size, h, w, c), dtype=np.float32)
                if i > 0:
                    min_h = min(att_batch.shape[1], h)
                    min_w = min(att_batch.shape[2], w)
                    min_c = min(att_batch.shape[3], c)
                    new_att[:i, :min_h, :min_w, :min_c] = att_batch[:i, :min_h, :min_w, :min_c]
                att_batch = new_att

            att_batch[i] = att_np

            info_struct = {}
            info_struct['id'] = self.ids[ri]
            info_struct['file_path'] = self.files[ri]
            infos.append(info_struct)

        # ensure defaults if no features were produced
        if fc_batch is None:
            fc_batch = np.zeros((batch_size, 2048), dtype=np.float32)
        if att_batch is None:
            att_batch = np.zeros((batch_size, 14, 14, 2048), dtype=np.float32)

        data = {}
        data['fc_feats'] = fc_batch
        # reshape att_feats to (batch_size, num_locations, feat_dim)
        H, W, C = att_batch.shape[1], att_batch.shape[2], att_batch.shape[3]
        data['att_feats'] = att_batch.reshape(batch_size, H * W, C)
        data['att_masks'] = None
        data['bounds'] = {'it_pos_now': self.iterator, 'it_max': self.N, 'wrapped': wrapped}
        data['infos'] = infos

        return data

    def reset_iterator(self, split):
        self.iterator = 0

    def get_vocab_size(self):
        return len(self.ix_to_word)

    def get_vocab(self):
        return self.ix_to_word