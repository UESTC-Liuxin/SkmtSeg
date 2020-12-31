# -*- coding: utf-8 -*-
"""
@description:

@author: LiuXin
@contact: xinliu1996@163.com
@Created on: 2020/12/30 下午4:16
"""

def get_inchannels(backbone):
    """

    :return:
    """
    if (backbone in ['resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnext50_32x4d', 'wide_resnet50_2', 'wide_resnet101_2']):
        in_channels = [2048,1024, 512, 256,64]
    elif backbone == 'mobilenet':
        in_channels = [320, 32, 24, 16]
    else:
        raise NotImplementedError

    return in_channels


def get_low_level_feat(backbone,inputs):
    if (backbone == 'xception'):  # 不同的backbone有不同的输出，处理不同
        low_level_feat = inputs[1]
    elif (backbone in ['resnet50', 'resnet101','wide_resnet50_2']):
        low_level_feat = inputs[3]
    else:
        NotImplementedError

    return low_level_feat
