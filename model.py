import math

import torch
import torch.nn.functional as F
from torch import nn

from resnet import resnet18, resnet50, seresnet50


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        m.eval()


class ProxyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProxyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        output = x.matmul(F.normalize(self.weight, dim=-1).t())
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)


class Model(nn.Module):
    def __init__(self, backbone_type, feature_dim, num_classes):
        super().__init__()

        # Backbone Network
        backbones = {'resnet18': (resnet18, 1), 'resnet50': (resnet50, 4), 'seresnet50': (seresnet50, 4)}
        backbone, expansion = backbones[backbone_type]
        self.layer0 = []
        for name, module in backbone(pretrained=True).named_children():
            if isinstance(module, nn.AdaptiveAvgPool2d) or isinstance(module, nn.Linear):
                continue
            if name not in ['layer1', 'layer2', 'layer3', 'layer4']:
                self.layer0.append(module)
            else:
                self.add_module(name, module)
        self.layer0 = nn.Sequential(*self.layer0)

        # Refactor Layer
        self.refactor = nn.Linear(512 * expansion, feature_dim, bias=False)
        # Classification Layer
        self.fc = nn.Sequential(ProxyLinear(feature_dim, num_classes))

    def forward(self, x):
        res0 = self.layer0(x)
        res1 = self.layer1(res0)
        res2 = self.layer2(res1)
        res3 = self.layer3(res2)
        res4 = self.layer4(res3)
        global_feature = torch.flatten(F.adaptive_max_pool2d(res4, output_size=(1, 1)), start_dim=1)
        global_feature = F.layer_norm(global_feature, [global_feature.size(-1)])
        feature = F.normalize(self.refactor(global_feature), dim=-1)
        classes = self.fc(feature)
        return feature, classes
