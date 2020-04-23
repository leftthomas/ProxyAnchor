import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

from model import ProxyLinear


def plot(embeds, labels, fig_path='./example.pdf'):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create a sphere
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0 * pi:100j]
    x = r * sin(phi) * cos(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(phi)
    ax.plot_surface(
        x, y, z, rstride=1, cstride=1, color='w', alpha=0.3, linewidth=0)
    ax.scatter(embeds[:, 0], embeds[:, 1], embeds[:, 2], c=labels, s=20)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(fig_path)


class ToyModel(nn.Module):
    def __init__(self, num_classes=10):
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

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = torch.flatten(x, start_dim=1)
        feature = F.normalize(self.fc_projection(x), dim=-1)
        classes = self.fc_final(feature)
        return feature, classes


def main():
    train_ds = datasets.FashionMNIST(
        root='./data',
        train=True,
        transform=transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=(0.1307,), std=(0.3081,))]),
        download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_ds,
                                               batch_size=args.batch_size,
                                               shuffle=True)
    example_loader = torch.utils.data.DataLoader(dataset=train_ds,
                                                 batch_size=args.batch_size,
                                                 shuffle=False)

    os.makedirs('./figs', exist_ok=True)

    print('Training Baseline model....')
    model_baseline = train_baseline(train_loader)
    bl_embeds, bl_labels = get_embeds(model_baseline, example_loader)
    plot(bl_embeds, bl_labels, fig_path='./figs/baseline.png')
    print('Saved Baseline figure')

    del model_baseline, bl_embeds, bl_labels

    loss_types = ['cosface', 'sphereface', 'arcface']
    for loss_type in loss_types:
        print('Training {} model....'.format(loss_type))
        model_am = train_am(train_loader, loss_type)
        am_embeds, am_labels = get_embeds(model_am, example_loader)
        plot(am_embeds, am_labels, fig_path='./figs/{}.png'.format(loss_type))
        print('Saved {} figure'.format(loss_type))
        del model_am, am_embeds, am_labels


def train_baseline(train_loader):
    model = ToyModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_step = len(train_loader)
    for epoch in tqdm(range(args.num_epochs)):
        for i, (feats, labels) in enumerate(tqdm(train_loader)):
            feats = feats.to(device)
            labels = labels.to(device)
            out = model(feats)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print('Baseline: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, args.num_epochs, i + 1, total_step, loss.item()))
        if ((epoch + 1) % 8 == 0):
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / 4
    return model.cpu()


def train_am(train_loader, loss_type):
    model = ConvAngularPen(loss_type=loss_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_step = len(train_loader)
    for epoch in tqdm(range(args.num_epochs)):
        for i, (feats, labels) in enumerate(tqdm(train_loader)):
            feats = feats.to(device)
            labels = labels.to(device)
            loss = model(feats, labels=labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print('{}: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(loss_type, epoch + 1, args.num_epochs, i + 1, total_step, loss.item()))

        if ((epoch + 1) % 8 == 0):
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / 4

    return model.cpu()


def get_embeds(model, loader):
    model = model.to(device).eval()
    full_embeds = []
    full_labels = []
    with torch.no_grad():
        for i, (feats, labels) in enumerate(loader):
            feats = feats[:100].to(device)
            full_labels.append(labels[:100].cpu().detach().numpy())
            embeds = model(feats, embed=True)
            full_embeds.append(F.normalize(embeds.detach().cpu()).numpy())
    model = model.cpu()
    return np.concatenate(full_embeds), np.concatenate(full_labels)


def parse_args():
    parser = argparse.ArgumentParser(description='Run Angular Penalty and Baseline experiments in fMNIST')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='input batch size for training (default: 512)')
    parser.add_argument('--num-epochs', type=int, default=40,
                        help='Number of epochs to train each model for (default: 20)')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed (default: 1234)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)
    main()
