#! python3.6
# -*- coding:utf-8 -*-
__author__ = 'bruce'
__time__ = '2019-08-10'
import torch
from torch import nn
import torch.nn.functional as F
from config import DefaultConfig


class BinaryFocalLoss(nn.Module):
    """针对二分类任务的Focal Loss"""
    def __init__(self, alpha=0.25, gamma=2, size_average=True):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, pred, target):
        """
        如果模型没有nn.Sigmoid()，那么这了就需要对预测结果计算一次Sigmoid操作
        :param pred: 预测的结果,shape为（batch_size, 1）
        :param target: 实际结果，shape为(batch_size 1)
        :return:
        """
        # pred = nn.Sigmoid()(pred)
        pred = pred.view(-1, 1)
        target = target.view(-1, 1)

        # 此处将预测样本为正负的概率都计算出来，此时pred.size=(batch_size, 2)
        pred = torch.cat((1-pred, pred), dim=1)

        # 根据target生成mask，即根据ground truth选择所需概率
        # 当标签为1时，我们就将模型预测该样本为正类的概率代入公式中进行计算
        # 当标签为0时，我们就将模型预测该样本为负类的概率代入公式中进行计算
        class_mask = torch.zeros(pred.shape[0], pred.shape[1])
        class_mask.scatter_(1, target.view(-1, 1).long(), 1.)

        # 利用mask将所需概率值挑选出来
        probs = (pred * class_mask).sum(dim=1).view(-1, 1)
        probs = probs.clamp(min=0.0001, max=1.0)

        # 计算概率的log值
        log_p = probs.log()

        # 对alpha进行设置（该参数用于调整正负样本数量不均衡带来的问题）
        alpha = torch.ones(pred.shape[0], pred.shape[1]).to(DefaultConfig.device)
        alpha[: 0] = alpha[:, 0] * (1 - self.alpha)
        alpha[:, 1] = alpha[:, 1] * self.alpha
        alpha = (alpha * class_mask).sum(dim=1).view(-1, 1)

        # 根据Focal Loss的公式计算Loss
        batch_loss = -alpha * (torch.pow((1-probs), self.gamma)) * log_p

        # Loss Function的常规操作，mean与sum的区别不大，相当于学习率设置不一样而已
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class FocalLossMultiLabel(nn.Module):
    """针对Multi-Label任务的Focal Loss"""
    def __init__(self, alpha=0.25, gamma=2, size_average=True):
        super(FocalLossMultiLabel, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, pred, target):
        criterion = BinaryFocalLoss(self.alpha, self.gamma, self.size_average)
        loss = torch.zeros(1, target.shape[1]).to(DefaultConfig.device)

        # 对每个Label计算一次Focal Loss
        for label in range(target.shape[1]):
            batch_loss = criterion(pred[:, label], target[:, label])
            loss[0, label] = batch_loss.mean()

        # Loss Function的常规操作
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()

        return loss
