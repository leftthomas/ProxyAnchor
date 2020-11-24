import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Model
from utils import recall, ImageReader, set_bn_eval, choose_loss

# for reproducibility
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(net, optim):
    net.train()
    # fix bn on backbone network
    net.backbone.apply(set_bn_eval)
    total_loss, total_num, features, targets, data_bar = 0.0, 0, [], [], tqdm(train_data_loader, dynamic_ncols=True)
    for inputs, labels in data_bar:
        inputs, labels = inputs.cuda(), labels.cuda()
        feature = net(inputs)
        loss = loss_func(feature, labels)
        optim.zero_grad()
        loss.backward()
        optim.step()

        if '*' in optimizer_type:
            # update weight
            if hasattr(loss_func, 'proxies'):
                proxies = loss_func.proxies.data
                updated_weight = proxies.index_select(0, labels) * (1.0 - momentum)
                proxies.index_copy_(0, labels, updated_weight)
                updated_feature = feature.detach() * momentum
                proxies.index_add_(0, labels, updated_feature)
            else:
                proxies = loss_func.W.data
                updated_weight = proxies.index_select(-1, labels) * (1.0 - momentum)
                proxies.index_copy_(-1, labels, updated_weight)
                updated_feature = feature.detach().t().contiguous() * momentum
                proxies.index_add_(-1, labels, updated_feature)

        features.append(F.normalize(feature.detach(), dim=-1))
        targets.append(labels)
        total_loss += loss.item() * inputs.size(0)
        total_num += inputs.size(0)
        data_bar.set_description('Train Epoch {}/{} - Loss:{:.4f}'.format(epoch, num_epochs, total_loss / total_num))

    features = torch.cat(features, dim=0)
    targets = torch.cat(targets, dim=0)
    data_base['train_features'] = features
    data_base['train_labels'] = targets
    if hasattr(loss_func, 'proxies'):
        data_base['train_proxies'] = F.normalize(loss_func.proxies.data, dim=-1)
    else:
        data_base['train_proxies'] = F.normalize(loss_func.W.data.t().contiguous(), dim=-1)
    return total_loss / total_num


def test(net, recall_ids):
    net.eval()
    with torch.no_grad():
        # obtain feature vectors for all data
        features = []
        for inputs, labels in tqdm(test_data_loader, desc='processing test data', dynamic_ncols=True):
            feature = net(inputs.cuda())
            features.append(F.normalize(feature, dim=-1))
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
    parser.add_argument('--loss_name', default='proxy_nca', type=str,
                        choices=['proxy_nca', 'normalized_softmax', 'cos_face', 'arc_face', 'proxy_anchor'],
                        help='loss name')
    parser.add_argument('--optimizer_type', default='adam*', type=str, choices=['adam*', 'sgd*', 'adam', 'sgd'],
                        help='optimizer type')
    parser.add_argument('--momentum', default=0.5, type=float, help='momentum used for the update of moving proxies')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--recalls', default='1,2,4,8', type=str, help='selected recall')
    parser.add_argument('--batch_size', default=64, type=int, help='training batch size')
    parser.add_argument('--num_epochs', default=20, type=int, help='training epoch number')

    opt = parser.parse_args()
    # args parse
    data_path, data_name, backbone_type, loss_name = opt.data_path, opt.data_name, opt.backbone_type, opt.loss_name
    optimizer_type, momentum, lr, batch_size = opt.optimizer_type, opt.momentum, opt.lr, opt.batch_size
    recalls, num_epochs = [int(k) for k in opt.recalls.split(',')], opt.num_epochs
    save_name_pre = '{}_{}_{}_{}_{}'.format(data_name, backbone_type, loss_name, optimizer_type, momentum)

    results = {'train_loss': []}
    for recall_id in recalls:
        results['test_recall@{}'.format(recall_id)] = []

    # dataset loader
    train_data_set = ImageReader(data_path, data_name, 'train', backbone_type)
    train_data_loader = DataLoader(train_data_set, batch_size, shuffle=True, num_workers=16)
    test_data_set = ImageReader(data_path, data_name, 'test', backbone_type)
    test_data_loader = DataLoader(test_data_set, batch_size, shuffle=False, num_workers=16)

    # model setup, optimizer config and loss definition
    model = Model(backbone_type).cuda()
    loss_func = choose_loss(loss_name, len(train_data_set.class_to_idx), 512).cuda()
    if '*' in optimizer_type:
        for param in loss_func.parameters():
            # not update by gradient
            param.requires_grad = False
    if 'adam' in optimizer_type:
        optimizer = Adam([{'params': model.parameters()}, {'params': loss_func.parameters()}], lr=lr)
    else:
        optimizer = SGD([{'params': model.parameters()}, {'params': loss_func.parameters()}], lr=lr)
    lr_scheduler = MultiStepLR(optimizer, milestones=[int(num_epochs * 0.5), int(num_epochs * 0.8)], gamma=0.1)

    data_base = {'test_images': test_data_set.images, 'test_labels': test_data_set.labels}
    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, optimizer)
        results['train_loss'].append(train_loss)
        test_features = test(model, recalls)
        lr_scheduler.step()

        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
        # save database and model
        data_base['test_features'] = test_features
        torch.save(model.state_dict(), 'results/{}_{}_model.pth'.format(save_name_pre, epoch))
        torch.save(data_base, 'results/{}_{}_data_base.pth'.format(save_name_pre, epoch))
