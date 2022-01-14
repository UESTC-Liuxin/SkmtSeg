# -*- coding: utf-8 -*-
"""
@description:

@author: LiuXin
@contact: xinliu1996@163.com
@Created on: 2020/10/27 下午5:31
"""


import torch
import torch.nn as nn
from modeling.head.section_cls import SectionClass


class SkmtNet(nn.Module):
    def __init__(self, backbone,auxiliary,trunk,num_classes,img_size):
        super(SkmtNet, self).__init__()
        self.backbone=backbone
        self.auxiliary=auxiliary
        self.trunk=trunk
        self.num_classes=num_classes

        self.section_cls=SectionClass("resnet50",nn.BatchNorm2d,16,img_size,5)

    def forward(self, input):
        img = input['image']
        base_out=self.backbone(img)
        section_out=self.section_cls(base_out)
        trunk_out = self.trunk(base_out)
        if(self.auxiliary):
            auxiliary_out=self.auxiliary(base_out)
        else:
            auxiliary_out=None

        return {'auxiliary_out':auxiliary_out,'trunk_out':trunk_out,'section_out':section_out}














