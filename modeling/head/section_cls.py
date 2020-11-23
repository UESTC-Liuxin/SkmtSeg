# -*- coding: utf-8 -*-
"""
@description:

@author: LiuXin
@contact: xinliu1996@163.com
@Created on: 2020/11/22 下午5:39
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SectionClass(nn.Module):
    def __init__(self,backbone,BatchNorm,output_stride,input_size,section_num_cls):
        super(SectionClass,self).__init__()
        self.backbone=backbone
        self.section_num_cls=section_num_cls
        if(backbone in ["resnet50","resnet101"]):
            in_channel=2048
        else:
            raise NotImplementedError
        self.Conv=nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=512,kernel_size=3,
                      padding=1),
            # BatchNorm(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        f=lambda x: x//(output_stride*2)
        self.linear1 = nn.Sequential(
            nn.Linear(512 * (f(input_size)**2),128),
            # nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(128,section_num_cls)
        )

    def forward(self, inputs):
        out=self.Conv(inputs[0])
        out = out.view(out.size()[0], -1)
        out=self.linear1(out)
        out=self.linear2(out)

        return out


if __name__ == '__main__':
    input=torch.rand(2,2048,32,32)
    model=SectionClass("resnet50",nn.BatchNorm2d,16,512,5)
    out=model([input,input])
    print(out.size())
