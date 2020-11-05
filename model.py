import torch.nn.functional as F
from pretrainedmodels import bninception
from torch import nn
from torchvision.models import resnet50, googlenet


class Model(nn.Module):
    def __init__(self, backbone_type, feature_dim):
        super().__init__()

        # Backbone Network
        backbones = {'resnet50': (resnet50, 2048), 'inception': (bninception, 1024), 'googlenet': (googlenet, 1024)}
        backbone, middle_dim = backbones[backbone_type]
        backbone = backbone(pretrained='imagenet' if backbone_type == 'inception' else True)
        if backbone_type == 'inception':
            backbone.global_pool = nn.AdaptiveMaxPool2d(1)
            backbone.last_linear = nn.Identity()
        else:
            backbone.avgpool = nn.AdaptiveMaxPool2d(1)
            backbone.fc = nn.Identity()
        self.backbone = backbone

        # Refactor Layer
        self.refactor = nn.Linear(middle_dim, feature_dim, bias=False)

    def forward(self, x):
        features = self.backbone(x)
        global_feature = F.layer_norm(features, features.size()[1:])
        feature = self.refactor(global_feature)
        return feature
