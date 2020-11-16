# -*- coding: utf-8 -*-
"""
@description:

@author: LiuXin
@contact: xinliu1996@163.com
@Created on: 2020/11/11 下午2:37
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class AuxiliaryFCN(nn.Module):

    def __init__(self, backbone:str,BatchNorm,output_stride,num_classes):

        super(AuxiliaryFCN, self).__init__()

        if(backbone=='mobilenet'):
            in_channel=320
        elif(backbone in ["resnet50","resnet101"]):
            in_channel=1024
        else:
            raise ValueError

        self.output_stride = output_stride
        self.last_conv = nn.Sequential(nn.Conv2d(in_channel, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
    def forward(self, inputs):
        x = inputs[1]  # 取倒数第二层
        x = self.last_conv(x)
        x = F.interpolate(x, scale_factor=self.output_stride, mode='bilinear', align_corners=True)

        return x


if __name__ == "__main__":
    model = AuxiliaryFCN(backbone='resnet50', BatchNorm=nn.BatchNorm2d,output_stride=16,num_classes=17)
    input = [torch.rand(2, 2048, 32, 32),torch.rand(2, 1024, 32, 32)]
    output = model(input)
    print(output.size())
