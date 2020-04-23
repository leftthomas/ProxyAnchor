import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from tqdm import tqdm

from model import ProxyLinear
from utils import LabelSmoothingCrossEntropyLoss

# for reproducibility
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(net, optim):
    net.train()
    total_loss, total_correct, total_num, data_bar = 0.0, 0.0, 0, tqdm(train_loader, dynamic_ncols=True)
    for inputs, labels in data_bar:
        inputs, labels = inputs.cuda(), labels.cuda()
        features, classes = net(inputs)
        loss = loss_criterion(classes, labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
        pred = torch.argmax(classes, dim=-1)
        total_loss += loss.item() * inputs.size(0)
        total_correct += torch.sum(pred == labels).item()
        total_num += inputs.size(0)
        data_bar.set_description('Train Epoch {}/{} - Loss:{:.4f} - Acc:{:.2f}%'
                                 .format(epoch, args.num_epochs, total_loss / total_num,
                                         total_correct / total_num * 100))

    return total_loss / total_num, total_correct / total_num * 100


def plot(embeds, labels, fig_path):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # create a sphere
    r, pi, cos, sin = 1, np.pi, np.cos, np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0 * pi:100j]
    x = r * sin(phi) * cos(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(phi)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color='w', alpha=0.3, linewidth=0)
    ax.scatter(embeds[:, 0], embeds[:, 1], embeds[:, 2], c=labels, s=20)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(fig_path)


class ToyModel(nn.Module):
    def __init__(self, loss_type, num_classes=10):
        super(ToyModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=8, stride=1))
        self.fc_projection = nn.Linear(512, 3)
        self.fc_final = ProxyLinear(3, num_classes)
        self.loss_type = loss_type

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = torch.flatten(x, start_dim=1)
        feature = F.normalize(self.fc_projection(x), dim=-1)
        if self.loss_type == 'ada':
            var, mean = torch.var_mean(feature, dim=0, unbiased=False, keepdim=True)
            classes = self.fc_final(((feature - mean) / torch.sqrt(var + 1e-5)))
        else:
            classes = self.fc_final(feature)
        return feature, classes


def test(net, data_loader):
    net.eval()
    full_features, full_labels = [], []
    with torch.no_grad():
        for inputs, labels in data_loader:
            features, classes = model(inputs.cuda())
            full_features.append(features.detach().cpu().numpy())
            full_labels.append(labels.detach().cpu().numpy())
    full_features, full_labels = np.concatenate(full_features), np.concatenate(full_labels)
    plot(full_features, full_labels, fig_path='{}_{}'.format(save_pre, 'toy_vis.pdf'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Toy example in fashionMNIST dataset')
    parser.add_argument('--batch_size', default=512, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=40, type=int, help='train epoch number')
    parser.add_argument('--loss_type', default='norm', type=str, choices=['norm', 'ada'], help='loss type')
    parser.add_argument('--temperature', default=1.0, type=float, help='temperature scale used in temperature softmax')
    args = parser.parse_args()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.286,), std=(0.353,))])
    train_data = FashionMNIST(root='data', train=True, download=True, transform=transform)
    test_data = FashionMNIST(root='data', train=False, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, num_workers=8)

    model = ToyModel(args.loss_type).cuda()
    optimizer = Adam(model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=args.num_epochs // 5, gamma=0.25)
    temperature = args.temperature if args.loss_type == 'norm' else 1.0
    loss_criterion = LabelSmoothingCrossEntropyLoss(temperature=temperature)
    save_pre = 'results/{}_{}'.format(args.loss_type, temperature)
    best_acc = 0.0
    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_accuracy = train(model, optimizer)
        lr_scheduler.step()
        if train_accuracy > best_acc:
            best_acc = train_accuracy
            torch.save(model.state_dict(), '{}_{}'.format(save_pre, 'toy_model.pth'))

    model.load_state_dict(torch.load('{}_{}'.format(save_pre, 'toy_model.pth'), map_location='cpu'))
    model = model.cuda()
    test(model, test_loader)
