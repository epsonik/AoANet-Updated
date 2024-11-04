import torch.nn as nn
import torch.nn.functional as F


class myDensenet(nn.Module):
    def __init__(self, densenet):
        super(myDensenet, self).__init__()
        self.densenet = densenet

    def forward(self, img, att_size=14):
        x = img.unsqueeze(0)
        x = self.densenet.model.features(x)
        fc = x.mean(3).mean(2).squeeze()
        att = F.adaptive_avg_pool2d(x, [att_size, att_size]).squeeze().permute(1, 2, 0)

        return fc, att
