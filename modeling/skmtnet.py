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
    def __init__(self, backbone,auxiliary,trunk1,trunk2,trunk3,trunk4,trunk5,num_classes):
        super(SkmtNet, self).__init__()
        self.backbone=backbone
        self.auxiliary=auxiliary
        #TODO:修改单个trunk为多个trunk
        self.trunk1=trunk1
        self.trunk2 = trunk2
        self.trunk3 = trunk3
        self.trunk4 = trunk4
        self.trunk5 = trunk5
        self.trunks=[self.trunk1,self.trunk2,self.trunk3,self.trunk4,self.trunk5]

        self.num_classes=num_classes

    def section_to_trunk(self,section):
        """

        :param section:
        :return:
        """
        return section





    def forward(self, input):
        img = input['image']
        sections= input['section']-1
        base_out=self.backbone(img)
        #提取单个batch的相关tensor出来组成新的列表
        trunk_out=self.trunks[sections[0]](base_out)
        # trunk_out = self.trunk(base_out)
        if(self.auxiliary):
            auxiliary_out=self.auxiliary(base_out)
        else:
            auxiliary_out=None

        return {'auxiliary_out':auxiliary_out,'trunk_out':trunk_out}














