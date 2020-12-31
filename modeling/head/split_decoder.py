# -*- coding: utf-8 -*-
"""
@description:

@author: LiuXin
@contact: xinliu1996@163.com
@Created on: 2020/12/30 上午9:41
"""
import torch
import torch.nn as nn


import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.model_utils.aspp import build_aspp
from modeling.model_utils.decoder import build_decoder
from modeling.model_utils.backbone2head import get_inchannels,get_low_level_feat
from modeling.head.danet import DANetHead


class SplitDeepLabDANet(nn.Module):
    def __init__(self, backbone,BatchNorm, output_stride, num_classes,freeze_bn=False):
        super(SplitDeepLabDANet, self).__init__()
        self.backbone=backbone
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder1 = build_decoder(num_classes, backbone, BatchNorm)
        self.decoder2 = build_decoder(num_classes, backbone, BatchNorm)
        self.decoder3 = build_decoder(num_classes, backbone, BatchNorm)
        self.decoder4 = build_decoder(num_classes, backbone, BatchNorm)
        self.decoder5 = build_decoder(num_classes, backbone, BatchNorm)
        self.decoders = [self.decoder1,self.decoder2,self.decoder3,self.decoder4,self.decoder5]
        self.output_stride = output_stride

        self.backbone = backbone
        in_channels=get_inchannels(self.backbone)
        self.head = DANetHead(in_channels[0], num_classes, BatchNorm)
        
        self.output_stride = output_stride
        if freeze_bn:
            self.freeze_bn()

    def forward(self, inputs,index):
        x0 = self.head(inputs[0])
        x = self.aspp(inputs[0])
        low_level_feat=get_low_level_feat(self.backbone,inputs)
        x = self.decoders[index](x, low_level_feat)
        x0 = F.interpolate(x0, scale_factor=8, mode='bilinear', align_corners=True)
        x = x0+x
        x = F.interpolate(x, scale_factor=self.output_stride/8, mode='bilinear', align_corners=True)
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p





