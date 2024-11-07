import torch.nn as nn
import torchvision


class DenseNet201(nn.Module):
    def __init__(self):
        super(DenseNet201, self).__init__()
        densenet201 = torchvision.models.densenet201(pretrained=True).cuda()

        modules = list(densenet201.children())[:-1]
        densenet201 = nn.Sequential(*modules)

        self.model = densenet201