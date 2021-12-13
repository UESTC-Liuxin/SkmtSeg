# -*- coding: utf-8 -*-
"""
@description:最基本的CrossEntropyLoss

@author: LiuXin
@contact: xinliu1996@163.com
@Created on: 2020/11/27 上午11:19
"""
import torch
import torch.nn as nn


class WCELoss(nn.Module):

    def __init__(self,#weight=[447433227,30668326,863040,56023299,36496764,4911119,5293877,1005469,5017141,700838,14504891],
                 weight=[0.22762148,0.22762148,0.91752577,0.22791293,0.22967742,1.11949686,0.99441341,0.72357724,0.71485944,1,0.71774194],
                 *args,**kwargs):
        super(WCELoss,self).__init__()
        if(weight):
            weight=1-torch.Tensor(weight).cuda()
        self.loss=nn.CrossEntropyLoss(weight,*args,**kwargs)

    def forward(self,logit, target):
        return self.loss(logit,target)


