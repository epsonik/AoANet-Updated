import os
import json
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from tqdm import tqdm
import numpy as np
import base64

# Custom RegNetX-16GF backbone with FPN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models import regnet_x_16gf

def build_regnet16_fpn():
    backbone = regnet_x_16gf(pretrained=True)
    return_layers = {'trunk_output': '0'}  # single output
    in_channels_stage2 = backbone.trunk_output[0][0].out_channels
    backbone = BackboneWithFPN(backbone.trunk_output, return_layers=return_layers,
                                in_channels_list=[in_channels_stage2], out_channels=256)
    return backbone

# Wrapper function for Faster R-CNN with RegNet16 backbone
def build_detector():
    backbone = build_regnet16_fpn()
    model = torchvision.models.detection.FasterRCNN(backbone, num_classes=91)  # 80 classes + background
    model.eval()
    return model

# Feature extraction for a dataset
def extract_features(model, dataloader, output_dir):
    os.makedirs(output_dir + '_att', exist_ok=True)
    os.makedirs(output_dir + '_fc', exist_ok=True)
    os.makedirs(output_dir + '_box', exist_ok=True)

    for images, targets in tqdm(dataloader):
        with torch.no_grad():
            outputs = model(images)

        for img, output in zip(targets, outputs):
            image_id = img["image_id"].item()
            boxes = output['boxes'].cpu().numpy()
            features = output['features'].cpu().numpy() if 'features' in output else np.zeros((len(boxes), 2048))

            np.savez_compressed(os.path.join(output_dir + '_att', str(image_id)), feat=features)
            np.save(os.path.join(output_dir + '_fc', str(image_id)), features.mean(0))
            np.save(os.path.join(output_dir + '_box', str(image_id)), boxes)

# COCO Dataset paths
coco_root = '/mnt/dysk2/dane/coco2014/'  # Replace with actual path
dataset_splits = {
    'train': ('train2014', 'annotations/instances_train2014.json'),
    'val': ('val2014', 'annotations/instances_val2014.json'),
    'test': ('val2014', 'annotations/image_info_test2014.json')  # test2014 does not have annotations
}

# Transform
transform = T.Compose([
    T.ToTensor(),
])

# Initialize model
model = build_detector()
model = model.cuda() if torch.cuda.is_available() else model

# Process each split
for split_name, (img_dir, ann_file) in dataset_splits.items():
    dataset = CocoDetection(root=os.path.join(coco_root, img_dir),
                            annFile=os.path.join(coco_root, ann_file),
                            transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
    extract_features(model, dataloader, output_dir=f'data/cocobu_{split_name}')
