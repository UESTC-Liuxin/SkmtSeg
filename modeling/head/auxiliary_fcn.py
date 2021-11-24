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
from modeling.model_utils.backbone2head import get_inchannels,get_low_level_feat

class AuxiliaryFCN(nn.Module):

    def __init__(self, backbone:str,BatchNorm,output_stride,num_classes):

        super(AuxiliaryFCN, self).__init__()

        in_channel=get_inchannels(backbone)[1]

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
        x = F.interpolate(x, scale_factor=self.output_stride/2, mode='bilinear', align_corners=True)

        return x


if __name__ == "__main__":
    model = AuxiliaryFCN(backbone='resnet50', BatchNorm=nn.BatchNorm2d,output_stride=16,num_classes=17)
    input = [torch.rand(2, 2048, 32, 32),torch.rand(2, 1024, 32, 32)]
    output = model(input)
    print(output.size())
