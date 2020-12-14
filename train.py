import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from adamp import AdamP
from torch.backends import cudnn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Model
from utils import recall, ImageReader, set_bn_eval, NormalizedSoftmaxLoss, ProxyAnchorLoss

# for reproducibility
np.random.seed(1)
torch.manual_seed(1)
cudnn.deterministic = True
cudnn.benchmark = False


def train(net, optim):
    net.train()
    # fix bn on backbone network
    net.backbone.apply(set_bn_eval)
    total_loss, total_correct, total_num, features, targets = 0.0, 0.0, 0, [], []
    data_bar = tqdm(train_data_loader, dynamic_ncols=True)
    for inputs, labels in data_bar:
        inputs, labels = inputs.cuda(), labels.cuda()
        feature, output = net(inputs)
        loss = loss_criterion(output, labels)
        optim.zero_grad()

        # handle the grad passed to proxies
        def hook_fn(grad):
            with torch.no_grad():
                if loss_name == 'proxy_anchor*':
                    weight = torch.exp(- loss_criterion.scale * (output - loss_criterion.margin))
                    pos_label = F.one_hot(labels, num_classes=output.size(-1))
                    pos_num = torch.sum(torch.ne(pos_label.sum(dim=0), 0))
                    # pos_weight = (torch.where(torch.eq(pos_label, 1), pos_output, torch.zeros_like(pos_output))).sum(
                    #     dim=0)
                    return grad
                elif loss_name == 'normalized_softmax*':
                    weight = (F.softmax(output * loss_criterion.scale, dim=-1) - 1) * loss_criterion.scale
                    pos_label = F.one_hot(labels, num_classes=output.size(-1))
                    pos_weight = torch.where(torch.eq(pos_label, 1), weight, torch.zeros_like(weight))
                    grad = pos_weight.t().mm(feature)
                    count = pos_label.sum(dim=0)
                    count = torch.where(torch.ne(count, 0), count, torch.ones_like(count))
                    grad = grad / count.unsqueeze(dim=-1)
                    return grad
                else:
                    return grad

        net.fc.weight.register_hook(hook_fn)
        loss.backward()
        optim.step()

        with torch.no_grad():
            features.append(feature)
            targets.append(labels)
            pred = torch.argmax(output, dim=-1)
            total_loss += loss.item() * inputs.size(0)
            total_correct += torch.sum(torch.eq(pred, labels)).item()
            total_num += inputs.size(0)
            data_bar.set_description('Train Epoch {}/{} - Loss:{:.4f} - Acc:{:.2f}%'
                                     .format(epoch, num_epochs, total_loss / total_num,
                                             total_correct / total_num * 100))

    features = torch.cat(features, dim=0)
    targets = torch.cat(targets, dim=0)
    data_base['train_features'] = features
    data_base['train_labels'] = targets
    data_base['train_proxies'] = F.normalize(net.fc.weight.data, dim=-1)
    return total_loss / total_num, total_correct / total_num * 100


def test(net, recall_ids):
    net.eval()
    # obtain feature vectors for all data
    with torch.no_grad():
        features = []
        for inputs, labels in tqdm(test_data_loader, desc='processing test data', dynamic_ncols=True):
            feature, _ = net(inputs.cuda())
            features.append(feature)
        features = torch.cat(features, dim=0)
        # compute recall metric
        acc_list = recall(features, test_data_set.labels, recall_ids)
        desc = 'Test Epoch {}/{} '.format(epoch, num_epochs)
        for index, rank_id in enumerate(recall_ids):
            desc += 'R@{}:{:.2f}% '.format(rank_id, acc_list[index] * 100)
            results['test_recall@{}'.format(rank_id)].append(acc_list[index] * 100)
        print(desc)
    return features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('--data_path', default='/home/data', type=str, help='datasets path')
    parser.add_argument('--data_name', default='car', type=str, choices=['car', 'cub'], help='dataset name')
    parser.add_argument('--backbone_type', default='resnet50', type=str, choices=['resnet50', 'inception', 'googlenet'],
                        help='backbone network type')
    parser.add_argument('--loss_name', default='proxy_anchor*', type=str,
                        choices=['proxy_anchor*', 'normalized_softmax*', 'proxy_anchor', 'normalized_softmax'],
                        help='loss name')
    parser.add_argument('--feature_dim', default=512, type=int, help='feature dim')
    parser.add_argument('--batch_size', default=64, type=int, help='training batch size')
    parser.add_argument('--num_epochs', default=20, type=int, help='training epoch number')
    parser.add_argument('--warm_up', default=2, type=int, help='warm up number')
    parser.add_argument('--recalls', default='1,2,4,8', type=str, help='selected recall')

    opt = parser.parse_args()
    # args parse
    data_path, data_name, backbone_type, loss_name = opt.data_path, opt.data_name, opt.backbone_type, opt.loss_name
    feature_dim, batch_size, num_epochs = opt.feature_dim, opt.batch_size, opt.num_epochs
    warm_up, recalls = opt.warm_up, [int(k) for k in opt.recalls.split(',')]
    save_name_pre = '{}_{}_{}_{}'.format(data_name, backbone_type, loss_name, feature_dim)

    results = {'train_loss': [], 'train_accuracy': []}
    for recall_id in recalls:
        results['test_recall@{}'.format(recall_id)] = []

    # dataset loader
    train_data_set = ImageReader(data_path, data_name, 'train', backbone_type)
    train_data_loader = DataLoader(train_data_set, batch_size, shuffle=True, num_workers=8)
    test_data_set = ImageReader(data_path, data_name, 'test', backbone_type)
    test_data_loader = DataLoader(test_data_set, batch_size, shuffle=False, num_workers=8)

    # model setup, optimizer config and loss definition
    model = Model(backbone_type, feature_dim, len(train_data_set.class_to_idx)).cuda()
    optimizer = AdamP([{'params': model.backbone.parameters()}, {'params': model.refactor.parameters()},
                       {'params': model.fc.parameters(), 'lr': 1e-2}], lr=1e-4)
    lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    if 'proxy_anchor' in loss_name:
        loss_criterion = ProxyAnchorLoss()
    else:
        loss_criterion = NormalizedSoftmaxLoss()

    data_base = {'test_images': test_data_set.images, 'test_labels': test_data_set.labels}
    for epoch in range(1, num_epochs + 1):
        # warmup, not update the parameters of backbone
        for param in model.backbone.parameters():
            param.requires_grad = False if epoch <= warm_up else True

        train_loss, train_accuracy = train(model, optimizer)
        results['train_loss'].append(train_loss)
        results['train_accuracy'].append(train_accuracy)
        test_features = test(model, recalls)
        lr_scheduler.step()

        # save database and model
        data_base['test_features'] = test_features
        torch.save(model.state_dict(), 'results/{}_{}_model.pth'.format(save_name_pre, epoch))
        torch.save(data_base, 'results/{}_{}_data_base.pth'.format(save_name_pre, epoch))
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
