import argparse

import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Model
from utils import recall, ImageReader


def train(net, optim):
    net.train()
    total_loss, total_correct, total_num, data_bar = 0.0, 0.0, 0, tqdm(train_data_loader, dynamic_ncols=True)
    for inputs, labels in data_bar:
        inputs, labels = inputs.cuda(), labels.cuda()
        features, classes = net(inputs)
        loss = loss_criterion(classes / temperature, labels)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(net.parameters(), 10)
        optim.step()
        pred = torch.argmax(classes, dim=-1)
        total_loss += loss.item() * inputs.size(0)
        total_correct += torch.sum(pred == labels).item()
        total_num += inputs.size(0)
        data_bar.set_description('Train Epoch {}/{} - Loss:{:.4f} - Acc:{:.2f}%'
                                 .format(epoch, num_epochs, total_loss / total_num, total_correct / total_num * 100))

    return total_loss / total_num, total_correct / total_num * 100


def test(net, recall_ids):
    net.eval()
    with torch.no_grad():
        # obtain feature vectors for all data
        for key in eval_dict.keys():
            eval_dict[key]['features'] = []
            for inputs, labels in tqdm(eval_dict[key]['data_loader'], desc='processing {} data'.format(key),
                                       dynamic_ncols=True):
                features, classes = net(inputs.cuda())
                eval_dict[key]['features'].append(features)
            eval_dict[key]['features'] = torch.cat(eval_dict[key]['features'], dim=0)

        test_features = torch.sign(eval_dict['test']['features'])
        # compute recall metric
        if data_name == 'isc':
            dense_acc_list = recall(eval_dict['test']['features'], test_data_set.labels, recall_ids,
                                    eval_dict['gallery']['features'], gallery_data_set.labels)
            gallery_features = torch.sign(eval_dict['gallery']['features'])
            binary_acc_list = recall(test_features, test_data_set.labels, recall_ids,
                                     gallery_features, gallery_data_set.labels, binary=True)
        else:
            dense_acc_list = recall(eval_dict['test']['features'], test_data_set.labels, recall_ids)
            binary_acc_list = recall(test_features, test_data_set.labels, recall_ids, binary=True)
    desc = 'Test Epoch {}/{} '.format(epoch, num_epochs)
    for index, rank_id in enumerate(recall_ids):
        desc += 'R@{}:{:.2f}%[{:.2f}%] '.format(rank_id, dense_acc_list[index] * 100, binary_acc_list[index] * 100)
        results['test_dense_recall@{}'.format(rank_id)].append(dense_acc_list[index] * 100)
        results['test_binary_recall@{}'.format(rank_id)].append(binary_acc_list[index] * 100)
    print(desc)
    return dense_acc_list[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('--data_path', default='/home/data', type=str, help='datasets path')
    parser.add_argument('--data_name', default='car', type=str, choices=['car', 'cub', 'sop', 'isc'],
                        help='dataset name')
    parser.add_argument('--feature_dim', default=2048, type=int, help='feature dim')
    parser.add_argument('--temperature', default=9, type=float, help='temperature scale used in temperature softmax')
    parser.add_argument('--recalls', default='1,2,4,8', type=str, help='selected recall')
    parser.add_argument('--batch_size', default=32, type=int, help='training batch size')
    parser.add_argument('--num_epochs', default=40, type=int, help='training epoch number')

    opt = parser.parse_args()
    # args parse
    data_path, data_name, feature_dim, temperature = opt.data_path, opt.data_name, opt.feature_dim, opt.temperature
    batch_size, num_epochs, recalls = opt.batch_size, opt.num_epochs, [int(k) for k in opt.recalls.split(',')]
    save_name_pre = '{}_{}_{}'.format(data_name, feature_dim, temperature)

    results = {'train_loss': [], 'train_accuracy': []}
    for recall_id in recalls:
        results['test_dense_recall@{}'.format(recall_id)] = []
        results['test_binary_recall@{}'.format(recall_id)] = []

    # dataset loader
    train_data_set = ImageReader(data_path, data_name, 'train')
    train_data_loader = DataLoader(train_data_set, batch_size, shuffle=True, num_workers=16, drop_last=True)
    test_data_set = ImageReader(data_path, data_name, 'query' if data_name == 'isc' else 'test')
    test_data_loader = DataLoader(test_data_set, batch_size, shuffle=False, num_workers=16)
    eval_dict = {'test': {'data_loader': test_data_loader}}
    if data_name == 'isc':
        gallery_data_set = ImageReader(data_path, data_name, 'gallery')
        gallery_data_loader = DataLoader(gallery_data_set, batch_size, shuffle=False, num_workers=16)
        eval_dict['gallery'] = {'data_loader': gallery_data_loader}

    # model setup, optimizer config and loss definition
    model = Model(feature_dim, len(train_data_set.class_to_idx)).cuda()
    optimizer = Adam([{'params': model.features.parameters()},
                      {'params': model.refactor.parameters()},
                      {'params': model.fc.parameters(), 'lr': 4e2},
                      ], lr=4e-3, eps=1.0)
    lr_scheduler = StepLR(optimizer, step_size=num_epochs // 2, gamma=0.1)
    loss_criterion = CrossEntropyLoss()

    best_recall = 0.0
    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = train(model, optimizer)
        results['train_loss'].append(train_loss)
        results['train_accuracy'].append(train_accuracy)
        rank = test(model, recalls)
        lr_scheduler.step()

        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
        # save database and model
        data_base = {}
        if rank > best_recall:
            best_recall = rank
            data_base['test_images'] = test_data_set.images
            data_base['test_labels'] = test_data_set.labels
            data_base['test_features'] = eval_dict['test']['features']
            if data_name == 'isc':
                data_base['gallery_images'] = gallery_data_set.images
                data_base['gallery_labels'] = gallery_data_set.labels
                data_base['gallery_features'] = eval_dict['gallery']['features']
            torch.save(model.state_dict(), 'results/{}_model.pth'.format(save_name_pre))
            torch.save(data_base, 'results/{}_data_base.pth'.format(save_name_pre))
