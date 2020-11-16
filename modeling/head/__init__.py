# -*- coding: utf-8 -*-
"""
@description:

@author: LiuXin
@contact: xinliu1996@163.com
@Created on: 2020/11/10 上午10:42
"""
from modeling.head import fcn,deeplabv3
from modeling.head import auxiliary_fcn
def build_head(head,backbone,BatchNorm, output_stride, num_classes):

    if(head=="fcn"):
        return fcn.FCN(backbone,BatchNorm,output_stride,num_classes)
    elif(head=="deeplab"):
        return deeplabv3.DeepLab(backbone,BatchNorm, output_stride, num_classes)
    else:
        raise NotImplementedError


def build_auxiliary_head(head,backbone,BatchNorm, output_stride, num_classes):
    if(head is None):
        return None
    if(head=="fcn"):
        return auxiliary_fcn.AuxiliaryFCN(backbone,BatchNorm,output_stride,num_classes)
    else:
        raise NotImplementedError