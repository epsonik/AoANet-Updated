import json
import os
from os.path import join

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from models.AttModel import MyResnet, MyDensenet


class DataLoaderRaw(Dataset):

    def __init__(self, opt):
        self.opt = opt
        self.folder_path = opt.folder_path
        self.image_list = self.get_image_list()
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        print('DataLoaderRaw loading images from folder: ', self.folder_path)
        print('listing all images in directory', self.folder_path)

        self.files = [f for f in os.listdir(self.folder_path)
                      if os.path.isfile(join(self.folder_path, f))]

        print('DataLoaderRaw found ', len(self.files), ' images')

        if 'densenet' in self.opt.cnn_model:
            self.cnn = MyDensenet(self.opt.cnn_model, pretrained=True)
        else:
            self.cnn = MyResnet(self.opt.cnn_model, pretrained=True)

        self.cnn.to(torch.device('cpu'))
        self.cnn.eval()

    def get_image_list(self):
        return

    def get_image_path_list(self):
        return self.files

    def __getitem__(self, index):
        # Open image
        img_path = join(self.folder_path, self.files[index])

        try:
            img = self.transforms(Image.open(img_path).convert('RGB'))
        except:
            return None, None

        img = img.reshape((1, 3, 224, 224))

        with torch.no_grad():
            tmp_att = self.cnn(img)

        return tmp_att, img_path

    def __len__(self):
        return len(self.files)