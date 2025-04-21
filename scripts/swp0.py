# -*- coding: utf-8 -*-
# File              : swp0.py
# Author            : Joy
# Date              : 2024/05/09
# Last Modified Date: 2024/09/16
# Last Modified By  : Joy
# Reference         :
# ******************************************************

import os
import cv2
import time
import torch
import pandas as pd

# os.chdir("./script")
# os.getcwd()

from data_preprocessing_yawn import VideoDataset, TripletDataset
from torchvision import transforms

import torch.optim as optim
from embedding import MobileViT_Classifier,TripletNet
from triplet_loss import TripletLoss
from trainer import fit
from utils import compute_binary_result

DIR_NAME = r'D:/GBI/research/P2 MetaModel/FedSWP/data/raw_data/'
if __name__ == '__main__':
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToPILImage(),  # 将 NumPy 数组转换为 PIL 图像
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[323.05114644, 303.04910693, 290.92570102],
                             std=[214.17018063, 211.37729198, 212.12888803])  # 归一化
    ])
    # 创建数据集
    video_dir = 'videos_yawn_SWP0.csv'
    label_dir = '/home/intern/research/yawn_detection/data/raw_data/markers'
    dataset = VideoDataset(video_dir, label_dir, transform=transform)

    # 数据集
    frame, label = dataset[0]
    print(f"数据大小: {len(dataset)}, 正负样本比例: {pd.Series(dataset.labels).mean().round(5)}")
    print(f"帧形状: {frame.shape}")  # 应为 (C, H, W)
    print(f"标签: {label}")

    # 划分数据集
    train_ratio = 0.7
    test_ratio = 0.3

    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

    # 打印划分结果
    print(f"训练集大小: {len(train_set)}, 正负样本比例: {pd.Series(train_set.dataset.labels).mean().round(5)}")
    print(f"测试集大小: {len(test_set)}, 正负样本比例: {pd.Series(test_set.dataset.labels).mean().round(5)}")

    n_classes = 2
    frame, label = train_set[0]
    print(frame.shape)

    triplet_train_dataset = TripletDataset(train_set)
    triplet_test_dataset = TripletDataset(test_set)

    # 载入数据集
    batch_size = 256  # 64, 512
    cuda = torch.version.cuda
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {'num_workers': 0, 'pin_memory': False}
    kwargs.update({
        'batch_size': 512,
        'shuffle': True
    })
    train_loader = torch.utils.data.DataLoader(triplet_train_dataset, **kwargs)
    test_loader = torch.utils.data.DataLoader(triplet_test_dataset, **kwargs)

    # 参数设置
    model_path = './model/'
    argdict = {
        'epochs': 10,
        'n_patience': 15,
        'log_interval': 10,
        'margin': 1.0,
        'lr': 1e-3,
        'cuda': torch.cuda.is_available()
    }

    # 设置模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding_net = MobileViT_Classifier(num_classes=2).to(device)
    model = TripletNet(embedding_net)
    print(model)

    if cuda:
        model.cuda()

    # 加载预训练权重
    pretrained_path = 'trained_model_mv_xs_swp0.pth'
    checkpoint = torch.load(pretrained_path, map_location='cuda' if cuda else 'cpu')
    model.load_state_dict(checkpoint)

    # 训练
    loss_fn = TripletLoss(argdict['margin'])
    optimizer = optim.Adam(model.parameters(), lr=argdict['lr'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, **argdict)



