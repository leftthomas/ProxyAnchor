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
        normalized_weight = F.normalize(self.weight, dim=-1)
        output = x.matmul(normalized_weight.t())
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)


class Model(nn.Module):
    def __init__(self, backbone_type, feature_dim, num_classes):
        super().__init__()

        # Backbone Network
        backbones = {'resnet50': (resnet50, 4), 'seresnet50': (seresnet50, 4)}
        backbone, expansion = backbones[backbone_type.replace('*', '')]
        self.features = []
        for name, module in backbone(pretrained=True, remove_downsample='*' in backbone_type).named_children():
            if isinstance(module, nn.Linear):
                continue
            self.features.append(module)
        self.features = nn.Sequential(*self.features)

        # Refactor Layer
        self.refactor = nn.Linear(512 * expansion, feature_dim, bias=False)
        # Classification Layer
        self.fc = ProxyLinear(feature_dim, num_classes)

    def forward(self, x):
        features = torch.flatten(self.features(x), start_dim=1)
        global_feature = F.layer_norm(features, features.size()[1:])
        feature = F.normalize(self.refactor(global_feature), dim=-1)
        classes = self.fc(feature)
        return feature, classes
