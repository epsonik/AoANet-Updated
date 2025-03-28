import torch.nn as nn
import torchvision

class RegNet16(nn.Module):
    def __init__(self):
        super(RegNet16, self).__init__()
        regnet16 = torchvision.models.regnet_y_16gf(weights="IMAGENET1K_SWAG_E2E_V1").cuda()

        modules = list(regnet16.children())[:-2]
        regnet16 = nn.Sequential(*modules)

        self.model = regnet16