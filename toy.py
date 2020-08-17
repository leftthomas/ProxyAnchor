import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from utils import recall, set_bn_eval, STL10
from model import ToyModel

# for reproducibility
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def for_loop(net, mode=True):
    net.train(mode)
    if mode:
        # fix bn on backbone network
        net.backbone.apply(set_bn_eval)
    data_bar = tqdm(train_loader if mode else test_loader, dynamic_ncols=True)
    total_loss, total_correct, total_num, embeds, outputs = 0.0, 0.0, 0, [], []
    for inputs, labels in data_bar:
        inputs, labels = inputs.cuda(), labels.cuda()
        features, classes = net(inputs)
        if mode:
            loss = loss_criterion(classes / temperature, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update weight
            if not with_learnable_proxy:
                updated_weight = F.normalize(net.fc_final.weight.clone().detach(), dim=-1).index_select(0, labels) * (
                        1.0 - momentum)
                net.fc_final.weight.index_copy_(0, labels, updated_weight)
                updated_feature = features.clone().detach() * momentum
                net.fc_final.weight.index_add_(0, labels, updated_feature)

            pred = torch.argmax(classes, dim=-1)
            total_loss += loss.item() * inputs.size(0)
            total_correct += torch.sum(pred == labels).item()
            total_num += inputs.size(0)
            data_bar.set_description('Train Epoch {}/{} - Loss:{:.4f} - Acc:{:.2f}%'
                                     .format(epoch, num_epochs, total_loss / total_num,
                                             total_correct / total_num * 100))
        else:
            embeds.append(features.detach())
            outputs.append(labels)
            data_bar.set_description('generate embeds for test data...')
    if mode:
        return total_loss / total_num, total_correct / total_num * 100
    else:
        embeds, outputs, feature_dict = torch.cat(embeds, dim=0).cpu(), torch.cat(outputs, dim=0).cpu().tolist(), {}
        acc_list = recall(embeds, outputs, [1])
        print('Test Epoch {}/{} R@1:{:.2f}%'.format(epoch, num_epochs, acc_list[0] * 100))
        for feature, label in zip(embeds, outputs):
            if label not in feature_dict:
                feature_dict[label] = [feature]
            else:
                feature_dict[label].append(feature)

        return feature_dict, acc_list[0]


def plot(embeds, fig_path):
    labels = list(embeds.keys())
    assert len(labels) == 3
    for label, color, marker in zip(labels, ['r', 'g', 'b'], ['o', '*', '^']):
        embed = torch.stack(embeds[label], dim=0)
        cluster = F.normalize(embed.mean(dim=0), dim=-1)
        plt.plot([0, cluster.numpy()[0]], [0, cluster.numpy()[1]], color=color)
        plt.scatter(embed.numpy()[:, 0], embed.numpy()[:, 1], s=5, c=color, marker=marker)
    plt.xlim([-1.2, 1.2])
    plt.ylim([-1.2, 1.2])
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run toy experiments in STL10 dataset')
    parser.add_argument('--data_path', default='/home/data/stl10', type=str, help='datasets path')
    parser.add_argument('--temperature', default=0.03, type=float, help='temperature scale used in temperature softmax')
    parser.add_argument('--with_learnable_proxy', action='store_true', help='use learnable proxy or not')
    parser.add_argument('--momentum', default=0.5, type=float, help='momentum used for the update of moving proxies')
    parser.add_argument('--batch_size', type=int, default=128, help='training batch size')
    parser.add_argument('--num_epochs', type=int, default=30, help='training epoch number')
    args = parser.parse_args()

    data_path, temperature, batch_size, num_epochs = args.data_path, args.temperature, args.batch_size, args.num_epochs
    with_learnable_proxy, momentum = args.with_learnable_proxy, args.momentum
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_dataset = STL10(root=data_path, train=True, transform=train_transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_dataset = STL10(root=data_path, train=False, transform=test_transform, download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    model = ToyModel(len(train_dataset.classes), with_learnable_proxy).cuda()
    optimizer = Adam(model.parameters(), lr=1e-4)
    lr_scheduler = StepLR(optimizer, step_size=num_epochs // 2, gamma=0.1)
    loss_criterion = nn.CrossEntropyLoss()

    results = {'train_loss': [], 'train_accuracy': [], 'test_recall': []}
    save_name_pre = '{}_{}_{}'.format(temperature, momentum, with_learnable_proxy)
    best_recall = 0.0
    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = for_loop(model, True)
        results['train_loss'].append(train_loss)
        results['train_accuracy'].append(train_accuracy)
        embeds_dict, rank = for_loop(model, False)
        results['test_recall'].append(rank * 100)
        lr_scheduler.step()
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results/toy_{}_statistics.csv'.format(save_name_pre), index_label='epoch')
        # save model, embeds and plot embeds
        if rank > best_recall:
            best_recall = rank
            torch.save(model.state_dict(), 'results/toy_{}_model.pth'.format(save_name_pre))
            torch.save(embeds_dict, 'results/toy_{}_embeds.pth'.format(save_name_pre))
            plot(embeds_dict, fig_path='results/{}_plot.pdf'.format(save_name_pre))
