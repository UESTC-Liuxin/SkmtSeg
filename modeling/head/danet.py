###########################################################################
# Created by: CASIA IVA 
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################
from __future__ import division
import torch.nn as nn
import torch.nn.functional as F
from modeling.model_utils.da_att import PAM_Module
from modeling.model_utils.da_att import CAM_Module

class DANet(nn.Module):
    def __init__(self, backbone,BatchNorm, output_stride, num_classes,freeze_bn=False):
        super(DANet, self).__init__()
        self.backbone = backbone
        if(backbone in ["resnet50","resnet101"]):
            in_channels=2048
        else:
            raise NotImplementedError

        self.head = DANetHead(in_channels, num_classes, BatchNorm)
        self.output_stride = output_stride
        if freeze_bn:
            self.freeze_bn()

    def forward(self, inputs):
        x = self.head(inputs[0])
        x = F.interpolate(x, scale_factor=self.output_stride,mode="bilinear",align_corners=True)

        '''  x = list(x)
        x[0] = upsample(x[0], self.output_stride, **self._up_kwargs)
        x[1] = upsample(x[1], self.output_stride, **self._up_kwargs)
        x[2] = upsample(x[2], self.output_stride, **self._up_kwargs)

        outputs = [x[0]]
        outputs.append(x[1])
        outputs.append(x[2])
        return tuple(outputs)'''
        return x
        
class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        # self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
        #                            BatchNorm(inter_channels),
        #                            nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   BatchNorm(inter_channels),
                                   nn.ReLU())

        # self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        # self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
        #                            BatchNorm(inter_channels),
        #                            nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   BatchNorm(inter_channels),
                                   nn.ReLU())

        # self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

        # self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        # feat1 = self.conv5a(x)
        # sa_feat = self.sa(feat1)
        # sa_conv = self.conv51(sa_feat)
        # sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        # feat_sum = sa_conv+sc_conv
        #
        # sasc_output = self.conv8(feat_sum)

        '''output = [sasc_output]
        output.append(sa_output)
        output.append(sc_output)'''
        return sc_output

