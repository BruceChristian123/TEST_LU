#! python3.7
# -*- coding:utf-8 -*-
__author__ = 'bruce'
__time__ = '2019-07-31'
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from Model.basic_model import BasicModule

__all__ = [
    'DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet264'
]


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('denseblock{}'.format(i))
        ft_module_names.append('transition{}'.format(i))
    ft_module_names.append('norm5')
    ft_module_names.append('classifier')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))

        # BottleNeck层，Conv(1*1)(bn_size*growth_rate,在DenseNet论文中是4k)
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2',  nn.Conv3d( bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
                 drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, compression=0.5):
        num_output_features = int(compression * num_input_features)
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))

        # Transition Layer中加入compression压缩参数
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class FeatureBlock(nn.Sequential):
    """
    FeatureBlock模块，即DenseNet的输入层与第一个Dense Block模块之间的部分
    """
    def __init__(self, in_channels, out_channels):
        super(FeatureBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_module('conv0', nn.Conv3d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False))
        self.add_module('norm0', nn.BatchNorm3d(out_channels))
        self.add_module('relu0', nn.ReLU(inplace=True))
        self.add_module('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1))


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ClassificationBlock(nn.Sequential):
    """
    DenseNet中的全连接层
    """
    def __init__(self, in_channels, out_classes):
        super(ClassificationBlock, self).__init__()
        self.in_channels = in_channels
        self.out_classes = out_classes

        self.add_module('norm5', nn.BatchNorm3d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('pool', nn.AvgPool3d(kernel_size=7, stride=1))
        self.add_module('flatten', Flatten())
        self.add_module('Linear', nn.Linear(in_channels, out_classes))



class DenseNet(BasicModule):
    """Densenet-BC model class
    Args:
        growth_rate (int) - how many filters to add each layer (k in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, sample_size=112, sample_duration=16, in_channels=1, growth_rate=32, block_config=(6, 12, 24),
                 num_init_features=64, compression=0.5, bn_size=4, drop_rate=0, num_classes=2):

        super(DenseNet, self).__init__()

        self.sample_size = sample_size
        self.sample_duration = sample_duration

        # First convolution
        # self.features = nn.Sequential(
        #     OrderedDict([
        #         ('conv0',  nn.Conv3d(1, num_init_features, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)),
        #         ('norm0', nn.BatchNorm3d(num_init_features)),
        #         ('relu0', nn.ReLU(inplace=True)),
        #         ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        #     ]))

        # FeatureBlock即第一个卷积层
        features = FeatureBlock(in_channels, num_init_features)
        self.features = nn.Sequential()
        self.features.add_module("FeatureBlock", features)

        # Each DenseBlock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate)
            # self.features.add_module('denseblock%d' % (i + 1), block)
            dense_block_name = "Dense Block: " + str(i+1)
            self.features.add_module(dense_block_name, block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                # trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                trans = _Transition(num_input_features=num_features, compression=compression)
                # self.features.add_module('transition%d' % (i + 1), trans)
                transition_layer_name = "Transition Layer: " + str(i+1)
                self.features.add_module(transition_layer_name, trans)
                # num_features = num_features // 2
                num_features = int(num_features * compression)

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
                # m.weight = nn.init.xavier_normal
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
        # self.features.add_module('Classification', ClassificationBlock(num_features, num_classes))

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        last_duration = int(math.ceil(self.sample_duration / 16))
        last_size = int(math.floor(self.sample_size / 32))
        # out = F.avg_pool3d(
        #     out, kernel_size=(last_duration, last_size, last_size)).view(
        #         features.size(0), -1)
        out = F.avg_pool3d(out, kernel_size=3, stride=2).view(features.size(0), -1)
        out = self.classifier(out)
        return out


def densenet121(**kwargs):
    model = DenseNet(
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        **kwargs)
    return model


def densenet169(**kwargs):
    model = DenseNet(
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 32, 32),
        **kwargs)
    return model


def densenet201(**kwargs):
    model = DenseNet(
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 48, 32),
        **kwargs)
    return model


def densenet264(**kwargs):
    model = DenseNet(
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 64, 48),
        **kwargs)
    return model
