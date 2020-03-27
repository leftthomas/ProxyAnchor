import math

import torch
import torch.nn.functional as F
from timm.models.gluon_resnet import default_cfgs
from timm.models.helpers import load_pretrained
from timm.models.layers import SEModule
from timm.models.resnet import ResNet, Bottleneck
from torch import nn


class EResNet(ResNet):

    def __init__(self, block, layers, cardinality=1, base_width=64, norm_layer=nn.BatchNorm2d, global_pool='avg',
                 remove_downsample=False, block_args=None):
        super(EResNet, self).__init__(block, layers, cardinality=cardinality, base_width=base_width,
                                      norm_layer=norm_layer, global_pool=global_pool, block_args=block_args)
        if remove_downsample:
            block_args = block_args or dict()
            # remove downsample for stage4
            self.inplanes = 256 * block.expansion
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, norm_layer=norm_layer, **block_args)


def resnet50(pretrained=False, **kwargs):
    default_cfg = default_cfgs['gluon_resnet50_v1b']
    model = EResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg)
    return model


def seresnet50(pretrained=False, **kwargs):
    default_cfg = default_cfgs['gluon_seresnext50_32x4d']
    model = EResNet(Bottleneck, [3, 4, 6, 3], cardinality=32, base_width=4, block_args=dict(attn_layer=SEModule),
                    **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg)
    return model


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
    def __init__(self, backbone_type, feature_dim, num_classes, remove_downsample):
        super().__init__()

        # Backbone Network
        backbones = {'resnet50': (resnet50, 4), 'seresnet50': (seresnet50, 4)}
        backbone, expansion = backbones[backbone_type]
        self.features = []
        for name, module in backbone(pretrained=True, global_pool='max',
                                     remove_downsample=remove_downsample).named_children():
            if isinstance(module, nn.Linear):
                continue
            self.features.append(module)
        self.features = nn.Sequential(*self.features)

        # Refactor Layer
        self.refactor = nn.Conv2d(512 * expansion, feature_dim, 1, bias=False)
        # Classification Layer
        self.fc = nn.Sequential(nn.BatchNorm1d(feature_dim), ProxyLinear(feature_dim, num_classes))

    def forward(self, x):
        features = self.features(x)
        global_feature = F.layer_norm(features, features.size()[1:])
        feature = F.normalize(torch.flatten(self.refactor(global_feature), start_dim=1), dim=-1)
        classes = self.fc(feature)
        return feature, classes
