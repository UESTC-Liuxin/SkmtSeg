# -*- coding: utf-8 -*-
"""
@description:

@author: LiuXin
@contact: xinliu1996@163.com
@Created on: 2020/11/27 下午8:05
"""

import torch
import torch.nn as nn
import warnings

class ComposeLoss(nn.Module):
    """Dice loss of binary class
    Args:
        losses:the list of loss Moudle
        losses_weights:list of losses_weights
        reduction:"mean"
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self,losses:list,losses_weights:[float,list],reduction="mean"):
        super(ComposeLoss,self).__init__()
        self.losses=losses
        if(isinstance(losses_weights,float) or isinstance(losses_weights,int)): #如果参数是一个小数或者整数，进行扩维到与losse一致
            losses_weights=[losses_weights]*len(losses)
        if(abs(sum(losses_weights)-1)>0.001):
            warnings.warn("the sum of weights is not 1")

        self.losses_wights=losses_weights
        self.reduction=reduction

    def forward(self,pred,gt):
        """
        caculate the all losses,and add the by weight
        :param inputs:
        :return:
        """

        tloss=torch.Tensor([0]).requires_grad_(False).cuda()
        for i,loss_func in enumerate(self.losses):
            tloss+=loss_func(pred,gt)*self.losses_wights[i]

        if(self.reduction=="mean"):
            return tloss/len(self.losses)
        return tloss


class CaculateTLoss(nn.Module):
    def __init__(self,trunk_crition:nn.Module,auxiliary_criterion:nn.Module=None):
        super(CaculateTLoss,self).__init__()
        self.auxiliary_criterion=auxiliary_criterion
        self.trunk_crition=trunk_crition


    def forward(self,pred,gt):
        loss = self.trunk_crition(pred['trunk_out'], gt)
        if(self.auxiliary_criterion is not None):
            loss_aux=self.auxiliary_criterion(pred['auxiliary_out'],gt)
            loss+=loss_aux
        return loss









