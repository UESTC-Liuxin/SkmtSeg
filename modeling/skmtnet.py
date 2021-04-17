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
        self.trunk1= trunk1
        self.trunk2= trunk2
        self.trunk3= trunk3
        self.trunk4= trunk4
        self.trunk5 = trunk5
        self.trunks= [self.trunk1,self.trunk2,self.trunk3,self.trunk4,self.trunk5]
        self.num_classes=num_classes

    def section_to_trunk(self,section):
        """
        将切面划分为[1,5],[2],[4]分别输出3个分支,[3]切面暂时没有
        :param section:
        :return:
        """
        if(section==1 or section==5):
            index=None
        elif(section==2):
            index=0
        else:
            index=1

        # 直接返回实际的切面类型
        return section-1


    def forward(self, input):
        img = input['image']
        sections= input['section']-1
        base_out=self.backbone(img)
        index=self.section_to_trunk(sections[0])
        #选择trunk
        trunk_out_section = self.trunks[index](base_out)
        if(self.auxiliary):
            auxiliary_out = self.auxiliary(base_out)
        else:
            auxiliary_out=None
        return {'auxiliary_out':auxiliary_out,'trunk_out':trunk_out_section}














