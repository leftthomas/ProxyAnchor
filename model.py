import torch
import torch.nn.functional as F
from torch import nn

from resnet import resnet50, seresnet50


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
        nn.init.normal_(self.weight)

    def forward(self, x):
        output = x.matmul(F.normalize(self.weight, dim=-1).t())
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)


class GatePool(nn.Module):
    def __init__(self, channels, reduction=16):
        super(GatePool, self).__init__()
        reduction_channels = max(channels // reduction, 8)
        self.conv1 = nn.Conv2d(channels, reduction_channels, kernel_size=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(reduction_channels)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv2 = nn.Conv2d(reduction_channels, channels, kernel_size=1, padding=0, bias=True)
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 1.5)

    def forward(self, x):
        max_value = F.adaptive_max_pool2d(x, output_size=(1, 1))
        avg_value = F.adaptive_avg_pool2d(x, output_size=(1, 1))

        x = self.relu(self.bn(self.conv1(x)))
        gate = self.relu(torch.tanh(self.conv2(self.avg_pool(x))))
        output = torch.where(gate > 0, max_value, avg_value)
        return output


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

        # pool
        self.pool = GatePool(512 * expansion)

        # Refactor Layer
        self.refactor = nn.Conv1d(512 * expansion, feature_dim, 1, bias=False)
        nn.init.kaiming_uniform_(self.refactor.weight)
        # Classification Layer
        self.fc = ProxyLinear(feature_dim, num_classes)

    def forward(self, x):
        features = self.features(x)
        global_feature = torch.flatten(self.pool(features), start_dim=2)
        global_feature = torch.flatten(self.refactor(global_feature), start_dim=1)
        feature = F.normalize(F.layer_norm(global_feature, global_feature.size()[1:]), dim=-1)
        var, mean = torch.var_mean(feature, dim=0, unbiased=False, keepdim=True)
        classes = self.fc(((feature - mean) / torch.sqrt(var + 1e-5)))
        return feature, classes
