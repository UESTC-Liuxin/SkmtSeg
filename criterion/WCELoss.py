# -*- coding: utf-8 -*-
"""
@description:最基本的CrossEntropyLoss

@author: LiuXin
@contact: xinliu1996@163.com
@Created on: 2020/11/27 上午11:19
"""
import torch
import torch.nn as nn
import numpy as np


class WCELoss(nn.Module):

    def __init__(self,weight=None,*args,**kwargs):
        super(WCELoss,self).__init__()
        if(weight):
            weight=torch.softmax(torch.FloatTensor(np.array(weight)), dim=0).cuda()
        self.loss=nn.CrossEntropyLoss(weight,*args,**kwargs)

    def forward(self,logit, target):
        return self.loss(logit,target)


