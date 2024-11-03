import torch.nn as nn
import torchvision.models.densenet

class DenseNet(torchvision.models.densenet.DenseNet):
    def __init__(self, block, layers, num_classes=64):
        super(DenseNet, self).__init__(block, layers, num_classes)
        maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0,
                                    ceil_mode=True) # change
        getattr(self,'features')['pool0'] = maxpool
        # fine tune 2, 3, 4
        for i in range(2, 4):
            getattr(self.features, 'denseblock%d'%i).conv1.stride = (2,2)
            getattr(self.features, 'denseblock%d'%i).conv2.stride = (1,1)

def densenet121(pretrained=False):
    model = DenseNet(32, (6, 12, 24, 16))
    return model

