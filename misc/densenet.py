import torch.nn as nn
import torchvision


class DenseNet121(nn.Module):
    def __init__(self):
        super(DenseNet121, self).__init__()
        densenet121 = torchvision.models.densenet121(pretrained=True).cuda()
        # maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0,
        #                        ceil_mode=True)  # change
        # getattr(densenet121, 'features').pool0 = maxpool
        #
        # # fine tune 2, 3, 4
        # for i in range(2, 5):
        #     getattr(densenet121.features, 'denseblock%d' % i).denselayer1.conv1.stride = (2, 2)
        #     getattr(densenet121.features, 'denseblock%d' % i).denselayer1.conv2.stride = (1, 1)

        modules = list(densenet121.children())[:-1]
        densenet121 = nn.Sequential(*modules)

        self.model = densenet121