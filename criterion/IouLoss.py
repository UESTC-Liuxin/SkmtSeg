# -*- coding: utf-8 -*-
"""
@description:

@author: LiuXin
@contact: xinliu1996@163.com
@Created on: 2020/11/27 下午7:16
"""

import torch
import torch.nn as nn
import torch.functional as F

class IouLoss(nn.Module):

    def __init__(self, n_classes):
        super(IouLoss, self).__init__()
        self.n_classes = n_classes

    @staticmethod
    def to_one_hot(tensor, n_classes):
        n, h, w = tensor.size()
        one_hot = torch.zeros(n, n_classes, h, w).scatter_(1, tensor.view(n, 1, h, w), 1)
        return one_hot

    def forward(self, pred, gt):
        # logit => N x Classes x H x W
        # target => N x H x W
        N = len(pred)
        pred = F.softmax(pred, dim=1)
        target_onehot = self.to_one_hot(gt, self.n_classes)
        # Numerator Product
        inter = pred * target_onehot
        # Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.n_classes, -1).sum(2)
        # Denominator
        union = pred + target_onehot - (pred * target_onehot)
        # Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.n_classes, -1).sum(2)
        loss = inter / (union + 1e-16)
        # Return average loss over classes and batch
        return -loss.mean()


