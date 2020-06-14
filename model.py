import torch
import torch.nn.functional as F
from torch import nn

from resnet import resnet50, seresnet50


class ProxyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProxyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # init proxy vector as unit random vector
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        output = x.matmul(F.normalize(self.weight, dim=-1).t())
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)


class Model(nn.Module):
    def __init__(self, backbone_type, feature_dim, num_classes, remove_common=True, use_temperature=False):
        super().__init__()

        # Backbone Network
        backbones = {'resnet50': (resnet50, 4), 'seresnet50': (seresnet50, 4)}
        backbone, expansion = backbones[backbone_type.replace('*', '')]
        self.features = []
        for name, module in backbone(pretrained=True, remove_downsample='*' in backbone_type).named_children():
            if isinstance(module, (nn.AdaptiveAvgPool2d, nn.Linear)):
                continue
            self.features.append(module)
        self.features = nn.Sequential(*self.features)
        self.pool = nn.AdaptiveMaxPool2d(output_size=1)

        # Refactor Layer
        self.refactor = nn.Linear(512 * expansion, feature_dim, bias=False)
        # Classification Layer
        self.fc = ProxyLinear(feature_dim, num_classes)

        self.remove_common = remove_common
        self.use_temperature = use_temperature

    def forward(self, x):
        features = self.features(x)
        global_feature = torch.flatten(self.pool(features), start_dim=1)
        global_feature = F.layer_norm(global_feature, global_feature.size()[1:])
        feature = F.normalize(self.refactor(global_feature), dim=-1)
        var, mean = torch.var_mean(feature, dim=0, unbiased=False, keepdim=True)
        if self.remove_common:
            if self.use_temperature:
                classes = self.fc(feature - mean)
            else:
                classes = self.fc((feature - mean) / torch.sqrt(var + 1e-5))
        else:
            if self.use_temperature:
                classes = self.fc(feature)
            else:
                classes = self.fc(feature / torch.sqrt(var + 1e-5))
        return feature, classes
