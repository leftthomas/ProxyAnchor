import torch
import torch.nn.functional as F
from torch import nn

from resnet import resnet50, seresnet50


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        m.eval()


class Model(nn.Module):
    def __init__(self, backbone_type, feature_dim, num_classes, remove_downsample):
        super().__init__()

        # Backbone Network
        backbones = {'resnet50': (resnet50, 4), 'seresnet50': (seresnet50, 4)}
        backbone, expansion = backbones[backbone_type]
        self.features = []
        for name, module in backbone(pretrained=True, remove_downsample=remove_downsample).named_children():
            if isinstance(module, (nn.AdaptiveAvgPool2d, nn.Linear)):
                continue
            self.features.append(module)
        self.features = nn.Sequential(*self.features)

        # Refactor Layer
        self.refactor = nn.Conv1d(512 * expansion, feature_dim, 1, bias=False)
        # Classification Layer
        self.fc = nn.Sequential(nn.BatchNorm1d(feature_dim), nn.Linear(feature_dim, num_classes, bias=False))

    def forward(self, x):
        features = self.features(x)
        global_feature = torch.flatten(F.adaptive_max_pool2d(features, output_size=(1, 1)), start_dim=2)
        global_feature = torch.flatten(self.refactor(global_feature), start_dim=1)
        feature = F.normalize(F.layer_norm(global_feature, global_feature.size()[1:]), dim=-1)
        classes = self.fc(feature)
        return feature, classes
