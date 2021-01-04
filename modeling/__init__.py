# -*- coding: utf-8 -*-
"""
@description:

@author: LiuXin
@contact: xinliu1996@163.com
@Created on: 2020/11/10 下午2:44
"""

import torch
import torch.nn as nn
from modeling.backbone import build_backbone
from modeling.head import build_head,build_auxiliary_head
from modeling.skmtnet import SkmtNet
from modeling.sync_batchnorm import SynchronizedBatchNorm1d,SynchronizedBatchNorm2d



def build_skmtnet(backbone:str,auxiliary_head,trunk_head,num_classes,output_stride=32,sync_bn=False):
    """
    :param backbone:the name of backbone
    :param auxiliary_head:
    :param trunk_head:
    :param num_classes:
    :param output_stride:
    :param sync_bn:
    :return:
    """
    #选择BN方式
    if sync_bn:
        BatchNorm=SynchronizedBatchNorm2d
    else:
        BatchNorm=nn.BatchNorm2d
    #选择backbone
    if (backbone):
        backbone_model = build_backbone(backbone, output_stride,BatchNorm,num_classes)
    else:
        backbone_model=None
    #选择auxiliary_head
    if(auxiliary_head):
        auxiliary_head_model=build_auxiliary_head(auxiliary_head,backbone,BatchNorm,output_stride,num_classes)
    else:
        auxiliary_head_model=None

    #选择trunk head
    trunk_head_model= build_head(trunk_head,backbone,BatchNorm,output_stride=output_stride,num_classes=num_classes)

    #集成模型
    return SkmtNet(backbone_model,auxiliary_head_model,trunk_head_model,num_classes)

class TempBatchNorm(nn.Module):
    def __init__(self,temp):
        super(TempBatchNorm,self).__init__()

    def forward(self, input):
        return input


if __name__ =="__main__":
    input =torch.Tensor(2,3,512,512).cuda()
    model=build_skmtnet('resnet50',auxiliary_head='fcn',trunk_head='deeplab_danet',
                        num_classes=11)
    model=model.cuda()
    out=model({'image':input,'section':torch.tensor([5,5])})


