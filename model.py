import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet50


class ProxyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProxyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # init proxy vector as unit random vector
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        output = F.normalize(x, dim=-1).matmul(F.normalize(self.weight, dim=-1).t())
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)


class Model(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()

        # Backbone Network
        self.features = []
        for name, module in resnet50(pretrained=True).named_children():
            if isinstance(module, (nn.AdaptiveAvgPool2d, nn.Linear)):
                continue
            self.features.append(module)
        self.features = nn.Sequential(*self.features)
        self.pool = nn.AdaptiveMaxPool2d(output_size=1)

        # Refactor Layer
        self.refactor = nn.Linear(2048, feature_dim)
        # Classification Layer
        self.fc = ProxyLinear(feature_dim, num_classes)

    def forward(self, x):
        features = self.features(x)
        global_feature = torch.flatten(self.pool(features), start_dim=1)
        global_feature = F.layer_norm(global_feature, global_feature.size()[1:])
        feature = self.refactor(global_feature)
        classes = self.fc(feature)
        return feature, classes
