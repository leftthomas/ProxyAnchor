import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

from model import ProxyLinear
from utils import LabelSmoothingCrossEntropyLoss, recall, obtain_density


# for reproducibility
# torch.manual_seed(0)
# np.random.seed(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


# use the first 7 classes as train classes, and the remaining classes as novel test classes
class FashionMNIST(datasets.FashionMNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        if train:
            self.classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt']
        else:
            self.classes = ['Sneaker', 'Bag', 'Ankle boot']
        super().__init__(root, train, transform, target_transform, download)
        datas, targets = [], []
        for data, target in zip(self.data, self.targets):
            if train:
                if target < 7:
                    datas.append(data)
                    targets.append(target)
            else:
                if target >= 7:
                    datas.append(data)
                    targets.append(target - 7)
        self.data, self.targets = torch.stack(datas, dim=0), torch.stack(targets, dim=0)


class ToyModel(nn.Module):
    def __init__(self, num_classes, use_temperature=False):
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
        self.fc_final = ProxyLinear(3, num_classes, use_temperature)

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


def for_loop(net, mode=True):
    net.train(mode)
    data_bar = tqdm(train_loader if mode else test_loader, dynamic_ncols=True)
    total_loss, total_correct, total_num = 0.0, 0.0, 0
    embeds, outputs = [], []
    for inputs, labels in data_bar:
        inputs, labels = inputs.cuda(), labels.cuda()
        features, classes = net(inputs)
        embeds.append(features.detach())
        outputs.append(labels)
        if mode:
            loss = loss_criterion(classes, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = torch.argmax(classes, dim=-1)
            total_loss += loss.item() * inputs.size(0)
            total_correct += torch.sum(pred == labels).item()
            total_num += inputs.size(0)
            data_bar.set_description('Train Epoch {}/{} - Loss:{:.4f} - Acc:{:.2f}%'
                                     .format(epoch, num_epochs, total_loss / total_num,
                                             total_correct / total_num * 100))
        else:
            data_bar.set_description('generate embeds for test data...')
    embeds, outputs = torch.cat(embeds, dim=0), torch.cat(outputs, dim=0).cpu().numpy()
    return embeds, outputs


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run temperature scale experiments in FashionMNIST')
    parser.add_argument('--data_path', default='/home/data/mnist', type=str, help='datasets path')
    parser.add_argument('--temperature', default=1.0, type=float, help='temperature scale used in temperature softmax')
    parser.add_argument('--batch_size', type=int, default=512, help='training batch size')
    parser.add_argument('--num_epochs', type=int, default=40, help='training epoch number')
    args = parser.parse_args()

    data_path, temperature, batch_size, num_epochs = args.data_path, args.temperature, args.batch_size, args.num_epochs
    data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))])
    train_ds = FashionMNIST(root=data_path, train=True, transform=data_transform, download=True)
    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
    test_ds = FashionMNIST(root=data_path, train=False, transform=data_transform, download=True)
    test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False, num_workers=8)

    model = ToyModel(len(train_ds.class_to_idx), temperature != 1.0).cuda()
    optimizer = Adam(model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=num_epochs // 5, gamma=0.25)
    loss_criterion = LabelSmoothingCrossEntropyLoss(temperature=temperature)

    for epoch in range(1, num_epochs + 1):
        # train
        embed, label = for_loop(model, True)
        acc_list = recall(embed, label, [1])
        desc = 'Train Epoch {}/{} R@1:{:.2f}% '.format(epoch, num_epochs, acc_list[0] * 100)
        density_list = obtain_density(embed, label)
        for key in sorted(list(density_list)):
            desc += 'D@{}:{:.2f}% '.format(key, density_list[key])
        print(desc)
        # plot(embed, label, fig_path='results/baseline.png')
        lr_scheduler.step()
        # test
        embed, label = for_loop(model, False)
        acc_list = recall(embed, label, [1])
        desc = 'Test Epoch {}/{} R@1:{:.2f}% '.format(epoch, num_epochs, acc_list[0] * 100)
        density_list = obtain_density(embed, label)
        for key in density_list:
            desc += 'D@{}:{:.2f}% '.format(key, density_list[key])
        print(desc)
        # plot(embed, label, fig_path='results/baseline.png')

        # save database and model
        # TODO
