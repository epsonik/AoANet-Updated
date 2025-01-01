import torch.nn as nn
import torchvision


class DenseNet121(nn.Module):
    def __init__(self):
        super(DenseNet121, self).__init__()
        densenet121 = torchvision.models.densenet121(pretrained=True).cuda()

        modules = list(densenet121.children())[:-1]
        densenet121 = nn.Sequential(*modules)

        self.model = densenet121