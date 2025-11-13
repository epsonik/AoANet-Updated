import torch
import torch.nn as nn
import os


def ResNet152(weights_path=None):
    """
    Load ResNet152 model with optional custom weights.
    
    Args:
        weights_path: Path to custom .pth weights file. If None or file doesn't exist,
                      uses default ImageNet weights.
    
    Returns:
        ResNet152 model ready to be used with myResnet wrapper
    """
    import torchvision.models as models
    
    # Initialize ResNet152 model
    if weights_path and os.path.exists(weights_path):
        # Load custom weights from .pth file
        resnet152 = models.resnet152(weights=None)
        state_dict = torch.load(weights_path, map_location='cpu')
        resnet152.load_state_dict(state_dict)
        resnet152 = resnet152.cuda()
    else:
        # Use default ImageNet weights
        resnet152 = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1).cuda()
    
    return resnet152
