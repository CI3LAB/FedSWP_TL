# -*- coding: utf-8 -*-
# File              : data_processing_yawn.py
# Author            : Joy
# Create Date       : 2024/03/11
# Last Modified Date: 2024/08/17
# Last Modified By  : Joy
# Reference         : frm / 255.0
# Description       : processes the YawDD dataset
# ******************************************************
import os
import random
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm


class VideoDataset(Dataset):
    def __init__(self, video_sets, label_dir, transform=None):
        """
        初始化数据集。

        参数:
            video_sets (str): 视频文件夹路径表的路径。
            label_dir (str): 标签文件夹路径。
            transform (callable, optional): 数据预处理函数。
        """
        self.video_dir = './yawn_detection/data/raw_data/videos'
        self.label_dir = label_dir
        self.transform = transform

        # 获取视频文件列表
        self.video_files = sorted(pd.read_csv(video_sets)['Name'])

        # 初始化帧和标签列表
        self.frame_paths = []
        self.labels = []

        # 遍历所有视频文件，加载帧和标签
        for video_file in tqdm(self.video_files, desc="加载视频"):
            # 构建标签文件名
            label_file = video_file.replace(".avi", "_mark.txt")
            label_path = os.path.join(self.label_dir, label_file)

            # 检查标签文件是否存在
            if not os.path.exists(label_path):
                # raise FileNotFoundError(f"标签文件不存在: {label_path}")
                print(f"标签文件不存在: {label_path}")
                continue

            # 加载标签
            with open(label_path, 'r') as f:
                labels = [int(line.strip()) for line in f.readlines()]  # 假设每行一个标签

            # 处理标签：将大于 1 的标签变为 1，0 保持不变
            labels = [0 if label == 0 else 1 for label in labels]

            # 将帧路径和标签添加到全局列表
            video_path = os.path.join(self.video_dir, video_file)
            self.frame_paths.append((video_path, len(labels)))  # 存储视频路径和帧数
            self.labels.extend(labels)

    def __len__(self):
        """返回数据集大小。"""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        获取单个样本。

        参数:
            idx (int): 样本索引。

        返回:
            frame (torch.Tensor): 视频帧张量，形状为 (C, H, W)。
            label (int): 标签。
        """
        # 找到对应的视频文件和帧索引
        frame_idx = idx
        for video_path, num_frames in self.frame_paths:
            if frame_idx < num_frames:
                break
            frame_idx -= num_frames

        # 使用 OpenCV 读取视频帧
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  # 跳转到指定帧
        ret, frame = cap.read()  # 读取帧
        cap.release()

        if not ret:
            raise ValueError(f"无法读取视频帧: {video_path}")

        # 将 BGR 转换为 RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)

        # 应用 transform
        if self.transform:
            frame = self.transform(frame)

        # 获取标签
        label = self.labels[idx]

        return frame, label


class TripletDataset(Dataset):
    def __init__(self, dataset):
        """
        初始化三元组数据集。

        参数:
            dataset (Dataset): 原始数据集。
        """
        self.dataset = dataset

        # 使用 tqdm 显示标签提取进度
        self.labels = []
        for _, label in dataset:
            self.labels.append(label)

        self.label_to_indices = self._build_label_to_indices()  # 构建标签到索引的映射

    #         self.progress_bar = tqdm(total=len(self.dataset), desc="生成三元组数据")  # 初始化进度条

    def _build_label_to_indices(self):
        """
        构建标签到索引的映射。

        返回:
            dict: 标签到索引列表的映射。
        """
        label_to_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)
        return label_to_indices

    def __len__(self):
        """返回数据集大小。"""
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        获取单个三元组样本。
        参数:idx (int): 样本索引。

        返回:
            anchor (torch.Tensor): 锚点样本。
            positive (torch.Tensor): 正样本。
            negative (torch.Tensor): 负样本。
        """
        # 获取锚点样本和标签
        anchor, anchor_label = self.dataset[idx]

        # 随机选择一个正样本（与锚点同一类别）
        positive_idx = random.choice(self.label_to_indices[anchor_label])
        positive, _ = self.dataset[positive_idx]

        # 随机选择一个负样本（与锚点不同类别）
        negative_label = random.choice(list(set(self.labels) - {anchor_label}))
        negative_idx = random.choice(self.label_to_indices[negative_label])
        negative, _ = self.dataset[negative_idx]

        #         # 更新进度条
        #         self.progress_bar.update(1)
        #         self.progress_bar.refresh()  # 刷新进度条显示

        return (anchor, positive, negative), []

    def __del__(self):
        """析构函数，关闭进度条。"""
        #         self.progress_bar.close()
        pass


##################### archived ######################
def preprocess_label(data, dir_name):
    """
    First, the existing labels on the different datasets are heterogeneous。
    mark range from 0 to 5; label range from [0, 1]
    mark:0 -> label:0
    mark:1,2,3,4,5  -> label:1

    :param data: pandas.core.frame.DataFrame, data = pd.read_csv("./data/videos_yawn_SWP1.csv")
    :param dir_name: str, dir_name = DIR_NAME+'markers/'
    :return: list
    """
    y = []
    # rates = []
    # check = []
    transform_rule = dict(zip(np.arange(6), [0, 1, 1, 1, 1, 1]))
    for i in range(len(data)):
        filename = data['Name'][i]
        file = dir_name + filename.replace(".avi", "_mark.txt")
        file = pd.read_csv(file, names=['yawn'])
        # check.append(len(file))
        file['relabel'] = file['yawn'].map(transform_rule)
        # rates.append(file['relabel'].sum())
        y.extend(file['relabel'].to_list())
    print("Fatigue rate: {:.2%}, Total Frames: {}".format(np.mean(y), len(y)))
    # data['FatigueNum'] = rates
    # data.to_csv(set_name)
    # y[:6506]+y[7054:]
    # frm[:6506]+frm[7067:]
    return y


def preprocess_frm(data, dir_name):
    """
    Second-a, videos to frames and reshape

    :param data: pandas.core.frame.DataFrame, data = pd.read_csv("./data/videos_yawn_SWP1.csv")
    :param dir_name: str, dir_name = DIR_NAME+"videos/"
    :return: a list of numpy.ndarray with shape (224, 224, 3) from (480, 640, 3)
    """
    frm = []
    lengths = []
    for i in range(len(data)):
        filename = data['Name'][i]
        if not os.path.exists(dir_name + filename):
            print('No path exists: ', filename)
        vin = cv2.VideoCapture(dir_name + filename)
        length = int(vin.get(cv2.CAP_PROP_FRAME_COUNT))
        lengths.append(length)
        for _ in range(0, length):
            ret, frame = vin.read()
            if ret: frm.append(cv2.resize(frame, (224, 224)))
    print("Shape: {}, Total Frames: {}".format(frm[0].shape, len(frm)))
    data['lengths'] = lengths
    # data.to_csv(set_name)
    return frm


def preprocess(data, frm, y):
    """
    Second, the videos from datasets consist of a range of frames.
    This technique selects a fixed number of frames representing the entire video,
    reducing computational costs and preserving important information.
    Then spilt into train and test sets.

    :param data: pandas.core.frame.DataFrame, data = pd.read_csv("./data/videos_yawn_SWP1.csv")
    :param frm: a list of numpy.ndarray with shape (224, 224, 3), if not TRAIN with shape (348, 3)
    :param y: list
    :return:
    """
    window_size = 10
    jump = 6
    sequential_data = []
    sequential_label = []
    n, N = 0, 0
    # for i in range(0, len(y), jump):
    for i in range(len(data)):
        n, N = N, N + data['lengths'][i]
        while (n + window_size) < N:
            sequential_data.append(frm[n:n + window_size])
            sequential_label.append(y[n + window_size])
            n += jump
        # print(len(sequential_label))
    assert len(sequential_data) == len(sequential_label)
    sequential_data, sequential_label = np.array(sequential_data), np.array(sequential_label)

    # Then spilt into train and test sets.
    if False:
        N = len(sequential_label)
        test_inds = np.random.choice(np.arange(N), int(N * 0.2), replace=False)
    else:
        test_inds = None
    print("Fatigue rate test: {:.2%}, with Input Shape {}, Frames train: {}, Frames test: {}".format(
        sequential_label[test_inds].mean(),
        sequential_data.shape,
        len(sequential_label) - len(test_inds),
        len(test_inds)))
    return sequential_data, sequential_label, test_inds


def get_triplets(y):
    """
    Third, the training samples are then formed into triplets.
    :param y: numpy.ndarray with shape (N,)
    :return: list of tuples
    """
    triplet_inds = []
    N = len(y)
    fatigue = np.arange(N)[y == 1]
    nonfatigue = np.arange(N)[y == 0]
    copies = 3
    for i, ind in enumerate(fatigue):
        pos = np.random.choice(np.delete(fatigue, i), copies, replace=False)
        neg = np.random.choice(nonfatigue, copies, replace=False)
        for j in range(copies): triplet_inds.append((ind, pos[j], neg[j]))
    copies = 2
    for i, ind in enumerate(nonfatigue):
        pos = np.random.choice(fatigue, copies, replace=False)
        neg = np.random.choice(np.delete(nonfatigue, i), copies, replace=False)
        for j in range(copies): triplet_inds.append((ind, pos[j], neg[j]))
    print("Fatigue rate train: {:.2%}, Triplets train: {}".format(y.mean(), len(triplet_inds)))
    np.random.shuffle(triplet_inds)
    return triplet_inds


class RawVideoDataset(Dataset):
    def __init__(self, video_sets, label_dir, transform=None):
        """
        初始化数据集。

        参数:
            video_dir (str): 视频文件夹路径。
            label_dir (str): 标签文件夹路径。
            transform (callable, optional): 数据预处理函数。
        """
        self.video_dir = '/home/intern/research/yawn_detection/data/raw_data/videos'
        self.label_dir = label_dir
        self.transform = transform

        # 获取视频文件列表
        self.video_files = sorted(pd.read_csv(video_sets)['Name'])  # sorted(os.listdir(video_dir))

        # 初始化帧和标签列表
        self.frame_paths = []
        self.labels = []

        # 遍历所有视频文件，加载帧和标签
        for video_file in tqdm(self.video_files, desc="加载视频"):
            # 构建标签文件名
            label_file = video_file.replace(".avi", "_mark.txt")
            label_path = os.path.join(self.label_dir, label_file)

            # 检查标签文件是否存在
            if not os.path.exists(label_path):
                # raise FileNotFoundError(f"标签文件不存在: {label_path}")
                print(f"标签文件不存在: {label_path}")
                continue

            # 加载标签
            with open(label_path, 'r') as f:
                labels = [int(line.strip()) for line in f.readlines()]  # 假设每行一个标签

            # 处理标签：将大于 1 的标签变为 1，0 保持不变
            labels = [0 if label == 0 else 1 for label in labels]

            # 将帧路径和标签添加到全局列表
            video_path = os.path.join(self.video_dir, video_file)
            self.frame_paths.append((video_path, len(labels)))  # 存储视频路径和帧数
            self.labels.extend(labels)

    def __len__(self):
        """返回数据集大小。"""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        获取单个样本。

        参数:
            idx (int): 样本索引。

        返回:
            frame (torch.Tensor): 视频帧张量，形状为 (C, H, W)。
            label (int): 标签。
        """
        # 找到对应的视频文件和帧索引
        frame_idx = idx
        for video_path, num_frames in self.frame_paths:
            if frame_idx < num_frames:
                break
            frame_idx -= num_frames

        # 使用 OpenCV 读取视频帧
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  # 跳转到指定帧
        ret, frame = cap.read()  # 读取帧
        cap.release()

        if not ret:
            raise ValueError(f"无法读取视频帧: {video_path}")

        # 将 BGR 转换为 RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 应用 transform
        if self.transform:
            frame = self.transform(frame)

        # 获取标签
        label = self.labels[idx]

        return frame, label
