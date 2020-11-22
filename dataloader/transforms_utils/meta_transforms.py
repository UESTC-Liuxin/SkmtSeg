# -*- coding: utf-8 -*-
"""
@description:

@author: LiuXin
@contact: xinliu1996@163.com
@Created on: 2020/11/20 上午11:42
"""
import torch

class MetaToTensor():
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        pass

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        sample['section']=torch.Tensor(sample)
        return sample