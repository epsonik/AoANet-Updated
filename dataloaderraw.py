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
    # trn.ToTensor(),
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

        # Dynamically import and initialize the specified CNN model
        cnn_model = opt.get('cnn_model', 'densenet161')

        if cnn_model == 'densenet121':
            from densenet_utils import myDensenet
            from densenet import DenseNet121
            net = DenseNet121()
            self.feature_size = 1024
            self.my_cnn = myDensenet(net)
        elif cnn_model == 'densenet161':
            from densenet_utils import myDensenet
            from densenet161 import DenseNet161
            net = DenseNet161()
            self.feature_size = 2208
            self.my_cnn = myDensenet(net)
        elif cnn_model == 'densenet169':
            from densenet_utils import myDensenet
            from densenet169 import DenseNet169
            net = DenseNet169()
            self.feature_size = 1664
            self.my_cnn = myDensenet(net)
        elif cnn_model == 'densenet201':
            from densenet_utils import myDensenet
            from densenet201 import DenseNet201
            net = DenseNet201()
            self.feature_size = 1920
            self.my_cnn = myDensenet(net)
        elif cnn_model == 'regnet':
            from densenet_utils import myDensenet
            from regnet import RegNet16
            net = RegNet16()
            self.feature_size = 3024
            self.my_cnn = myDensenet(net)
        elif cnn_model == 'inception':
            from densenet_utils import myDensenet
            from inception import Inception
            net = Inception()
            self.feature_size = 2048
            self.my_cnn = myDensenet(net)
        elif cnn_model == 'resnet101':
            from resnet_utils import myResnet
            from resnet import resnet101
            import torchvision.models as models
            # Load pretrained ResNet101
            net = models.resnet101(pretrained=True)
            self.feature_size = 2048
            self.my_cnn = myResnet(net)
        elif cnn_model == 'resnet152':
            from resnet_utils import myResnet
            from resnet import resnet152
            import torchvision.models as models
            # Load pretrained ResNet152
            net = models.resnet152(pretrained=True)
            self.feature_size = 2048
            self.my_cnn = myResnet(net)
        else:  # Default to densenet161
            from densenet_utils import myDensenet
            from densenet161 import DenseNet161
            net = DenseNet161()
            self.feature_size = 2208
            print(f"Warning: CNN model '{cnn_model}' not recognized. Defaulting to densenet161.")
            self.my_cnn = myDensenet(net)

        self.my_cnn.cuda()
        self.my_cnn.eval()

        # load the json file which contains additional information about the dataset
        print('DataLoaderRaw loading images from folder: ', self.folder_path)

        self.files = []
        self.ids = []

        print(len(self.coco_json))
        if len(self.coco_json) > 0:
            print('reading from ' + opt.coco_json)
            # read in filenames from the coco-style json file
            self.coco_annotation = json.load(open(self.coco_json))
            for k, v in enumerate(self.coco_annotation['images']):
                fullpath = os.path.join(self.folder_path, v['file_name'])
                self.files.append(fullpath)
                self.ids.append(v['id'])
        else:
            # read in all the filenames from the folder
            print('listing all images in directory ' + self.folder_path)

            def isImage(f):
                supportedExt = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM']
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
                        self.ids.append(str(n))  # just order them sequentially
                        n = n + 1

        self.N = len(self.files)
        print('DataLoaderRaw found ', self.N, ' images')

        self.iterator = 0

    def get_batch(self, split, batch_size=None):
        batch_size = batch_size or self.batch_size

        # pick an index of the datapoint to load next
        fc_batch = np.ndarray((batch_size, self.feature_size), dtype='float32')
        att_batch = np.ndarray((batch_size, 14, 14, self.feature_size), dtype='float32')
        max_index = self.N
        wrapped = False
        infos = []

        for i in range(batch_size):
            ri = self.iterator
            ri_next = ri + 1
            if ri_next >= max_index:
                ri_next = 0
                wrapped = True
                # wrap back around
            self.iterator = ri_next

            img = skimage.io.imread(self.files[ri])

            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
                img = np.concatenate((img, img, img), axis=2)

            img = img[:, :, :3].astype('float32') / 255.0
            img = torch.from_numpy(img.transpose([2, 0, 1])).cuda()
            img = preprocess(img)
            with torch.no_grad():
                tmp_fc, tmp_att = self.my_cnn(img)

            fc_batch[i] = tmp_fc.data.cpu().float().numpy()
            att_batch[i] = tmp_att.data.cpu().float().numpy()

            info_struct = {}
            info_struct['id'] = self.ids[ri]
            info_struct['file_path'] = self.files[ri]
            infos.append(info_struct)

        data = {}
        data['fc_feats'] = fc_batch
        data['att_feats'] = att_batch.reshape(batch_size, -1, self.feature_size)
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