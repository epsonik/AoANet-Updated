from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from captioning.models import AoAModel

import models
import vis_utils
from dataloader_raw import DataLoaderRaw
from models import AttModel

def main(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # load the data
    loader = DataLoaderRaw(opt)
    # when using fusion model, we need to load region features
    # (both image and region features are calculated in dataloader)
    loader.ix_to_word = loader.get_vocab()

    # Create the model
    infos = json.load(open(opt.infos_path))
    infos['opt'].vocab = loader.get_vocab()
    model = AoAModel(infos['opt'])

    print(f"Loading model from: {opt.model}")
    model.load_state_dict(torch.load(opt.model, map_location=torch.device('cpu')))
    print("Model loaded successfully")
    model.to(torch.device('cpu'))
    model.eval()

    print('start')

    img_list = loader.get_image_path_list()

    num_of_img = opt.num_images
    if int(num_of_img) == 0:
        num_of_img = len(img_list)

    print(f'Processing images from: {opt.image_folder}')
    print(f'Saving visualizations to: {opt.vis_path}')
    print(f"Number of images to process: {'all' if num_of_img == len(img_list) else num_of_img}\n")

    for i in range(num_of_img):
        print(f"Processing image {i + 1}: {loader.files[i]}")
        with torch.no_grad():
            tmp_att, img_path = loader.__getitem__(i)
            if tmp_att is None:
                continue
            tmp_fc = torch.from_numpy(np.mean(tmp_att.numpy(), axis=1))
            att_feats = tmp_att

            # Run the model
            seq, attention_weights = vis_utils.capture_attention_weights(
                model, tmp_fc, att_feats, None, opt, loader
            )

            # Create visualization
            vis_utils.create_visualizations(
                seq, attention_weights, img_path, loader, opt.vis_path
            )

    print("\nProcessing complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input json
    parser.add_argument('--model', type=str,
                        default='log_aoanet/model-best.pth',
                        help='path to model to evaluate')
    parser.add_argument('--infos_path', type=str,
                        default='log_aoanet/infos_aoanet-best.json',
                        help='path to infos to evaluate')

    # parser.add_argument('--cnn_model', type=str, default='densenet161',
    #                     help='resnet101, resnet152, densenet161')
    parser.add_argument('--cnn_model', type=str, default='resnet101',
                        help='resnet101, resnet152, densenet161')
    opt = parser.parse_args()

    # Set visualization path
    opt.vis_path = f"vis/{opt.cnn_model}_aoanet"
    if not os.path.exists(opt.vis_path):
        os.makedirs(opt.vis_path)

    # Set paths for the script
    opt.folder_path = opt.image_folder

    main(opt)
