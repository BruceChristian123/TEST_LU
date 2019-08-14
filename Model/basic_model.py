#! python3.6
# -*- coding:utf-8 -*-
__author__ = 'bruce'
__time__ = '2019-07-24'
import os
import time
from pathlib import Path
import torch as t
from torch import nn


class BasicModule(nn.Module):
    """
    封装nn.Module，主要提供save和load方法
    """
    def __init__(self, opt=None):
        super(BasicModule, self).__init__()
        # self.model_name = str(type(self))
        self.model_name = str(self.__class__.__name__)

    def load(self, path):
        """
        可加载指定路径的模型
        :param path:
        :return:
        """
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        """
        保存模型，默认使用"模型名字+时间"作为文件名，例如DenseNet_0723_20:50:29.pth
        :param name:
        :return:
        """
        if name is None:
            # prefix = 'checkpoints/' + self.model_name + '_'
            current_path = Path.cwd() / 'checkpoints'
            prefix = self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
            name = current_path / name
        t.save(self.state_dict(), str(name))
        return str(name)

    def get_optimizer(self, lr, weight_decay):
        return t.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)


class Flatten(nn.Module):
    """
    把输入reshape成(batch_size, dim_length)
    """
    def forward(self, x):
        return x.view(x.size(0), -1)