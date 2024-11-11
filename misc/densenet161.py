import torch.nn as nn
import torchvision


class DenseNet161(nn.Module):
    def __init__(self):
        super(DenseNet161, self).__init__()
        densenet161 = torchvision.models.densenet161(pretrained=True).cuda()
        modules = list(densenet161.children())[:-1]
        densenet161 = nn.Sequential(*modules)

        self.model = densenet161