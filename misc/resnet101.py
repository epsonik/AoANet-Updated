import torch
import torch.nn as nn
import os


def ResNet101(weights_path=None):
    """
    Load ResNet101 model with optional custom weights.
    
    Args:
        weights_path: Path to custom .pth weights file. If None or file doesn't exist,
                      uses default ImageNet weights.
    
    Returns:
        ResNet101 model ready to be used with myResnet wrapper
    """
    import torchvision.models as models
    
    # Initialize ResNet101 model
    if weights_path and os.path.exists(weights_path):
        # Load custom weights from .pth file
        resnet101 = models.resnet101(weights=None)
        state_dict = torch.load(weights_path, map_location='cpu')
        resnet101.load_state_dict(state_dict)
        resnet101 = resnet101.cuda()
    else:
        # Use default ImageNet weights
        resnet101 = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1).cuda()
    
    return resnet101
