import torch.nn as nn
from torchvision.models.resnet import Bottleneck, ResNet, conv3x3, conv1x1
from torchvision.models.utils import load_state_dict_from_url

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'seresnet50': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-cadene/se_resnet50-ce0d4300.pth'
}


class SEModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return identity * x


class SEBottleneck(Bottleneck):

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None):
        super(SEBottleneck, self).__init__(inplanes, planes, stride, downsample, groups, base_width, dilation,
                                           norm_layer)
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width, stride)
        self.conv2 = conv3x3(width, width, groups=groups, dilation=dilation)
        self.se_module = SEModule(planes * self.expansion)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.se_module(out) + identity
        out = self.relu(out)

        return out


class EResNet(ResNet):

    def __init__(self, block, layers, num_classes=1000, ceil_mode=False, remove_downsample=False):
        super(EResNet, self).__init__(block, layers, num_classes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=ceil_mode)
        if remove_downsample:
            # remove down sample for stage4
            self.inplanes = 256 * block.expansion
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = EResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress, map_location='cpu')
        if arch == 'seresnet50':
            for key in list(state_dict.keys()):
                if 'layer0' in key:
                    state_dict[key.replace('layer0.', '')] = state_dict.pop(key)
                if 'last_linear' in key:
                    state_dict[key.replace('last_linear', 'fc')] = state_dict.pop(key)
        model.load_state_dict(state_dict)
    return model


def resnet50(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def seresnet50(pretrained=False, progress=True, **kwargs):
    kwargs['ceil_mode'] = True
    return _resnet('seresnet50', SEBottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)