# -*- coding: utf-8 -*-
"""
@description:

@author: LiuXin
@contact: xinliu1996@163.com
@Created on: 2020/10/27 下午5:31
"""


import torch
import torch.nn as nn


class SkmtNet(nn.Module):
    def __init__(self, backbone,auxiliary,trunk,num_classes):
        super(SkmtNet, self).__init__()
        self.backbone=backbone
        self.auxiliary=auxiliary
        self.trunk=trunk
        self.num_classes=num_classes

    def forward(self, input):
        img = input['image']
        base_out=self.backbone(img)

        trunk_out = self.trunk(base_out)
        if(self.auxiliary):
            auxiliary_out=self.auxiliary(base_out)
        else:
            auxiliary_out=None

        return {'auxiliary_out':auxiliary_out,'trunk_out':trunk_out}














