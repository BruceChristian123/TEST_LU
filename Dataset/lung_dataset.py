#! python3.6
# -*- coding:utf-8 -*-
__author__ = 'bruce'
__time__ = '2019-08-10'
import os
from torch.utils import data
import numpy as np
import torch
from torchvision import transforms as T


class LungNodule(data.Dataset):
    def __init__(self, root, train=True, test=False):
        """
        获取数据集，并根据训练、验证和测试划分数据
        :param root: 数据存放的路径
        :param transforms:
        :param train: =True表示训练集
        :param test:  =True且train=False表示测试集
        """
        if train:
            src_path = os.path.join(root, 'train')
        elif test:
            src_path = os.path.join(root, 'test')
        else:
            src_path = os.path.join(root, 'val')
        self.src_path = src_path

    def __getitem__(self, idx):
        """
        每次返回一条数据
        :param idx: 索引
        :return:
        """
        fname = os.listdir(self.src_path)[idx]
        if '_aug' in fname:
            label = 1
        else:
            label = 0
        fname = os.path.join(self.src_path, fname)
        data = np.load(fname)
        depth, height, width = data.shape
        data = data.reshape((1, depth, height, width))
        return data, label

    def __len__(self):
        return len(os.listdir(self.src_path))
