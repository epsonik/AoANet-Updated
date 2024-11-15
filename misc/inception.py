import torch.nn as nn
import torchvision


class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()
        inception = torchvision.models.inception_v3(pretrained=True)
        inception.aux_logits = False
        inception.AuxLogits = None
        modules = list(inception.children())[:-3]
        inception = nn.Sequential(*modules)

        self.model = inception