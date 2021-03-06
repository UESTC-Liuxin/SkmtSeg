###########################################################################
# Created by: CASIA IVA 
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################
from __future__ import division
import torch.nn as nn
import torch
import torch.nn.functional as F
from modeling.model_utils.da_att import DANetHead


class MultiScaleAttention(nn.Module):
    def __init__(self, backbone,BatchNorm, output_stride, num_classes,freeze_bn=False):
        super(MultiScaleAttention, self).__init__()
        self.backbone = backbone
        if(backbone in ["resnet50","resnet101"]):
            in_channels=2048
        else:
            raise NotImplementedError
        self.down4 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU())
        self.down3 = nn.Sequential(nn.Conv2d(in_channels//2, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU())
        self.down2 = nn.Sequential(nn.Conv2d(in_channels//4, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU())
        self.down1 = nn.Sequential(nn.Conv2d(in_channels//8, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU())

        self.fuse1 = MultiConv(256, 64)

        self.head4 = DANetHead(in_channels, 64, BatchNorm)
        self.head3 = DANetHead(in_channels//2, 64, BatchNorm)
        self.head2 = DANetHead(in_channels//4, 64, BatchNorm)
        self.head1 = DANetHead(in_channels//8, 64, BatchNorm)

        # self.head4 = DANetHead(128, 64, BatchNorm)
        # self.head3 = DANetHead(128, 64, BatchNorm)
        # self.head2 = DANetHead(128, 64, BatchNorm)
        self.head = DANetHead(128, num_classes, BatchNorm)

        self.output_stride = output_stride

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, num_classes, 1))
        if freeze_bn:
            self.freeze_bn()

    def forward(self, inputs):
        # inputs4 = self.head4(inputs[0])
        # inputs3 = self.head3(inputs[1])
        # inputs2 = self.head2(inputs[2])

        inputs4 = self.down4(inputs[0])
        inputs3 = self.down3(inputs[1])
        inputs2 = self.down2(inputs[2])
        down4 = F.upsample(inputs4, size=inputs[3].size()[2:], mode='bilinear')
        down3 = F.upsample(inputs3, size=inputs[3].size()[2:], mode='bilinear')
        down2 = F.upsample(inputs2, size=inputs[3].size()[2:], mode='bilinear')
        down1 = self.down1(inputs[3])

        fuse1 = self.fuse1(torch.cat((down4, down3, down2, down1), 1))
        fuse1 = self.conv8(fuse1)
        #attention4 = self.head(torch.cat((down4, fuse1), dim=1))
        #attention3 = self.head3(torch.cat((down3, fuse1), dim=1))
        #attention2 = self.head2(torch.cat((down2, fuse1), dim=1))
        #attention1 = self.head1(torch.cat((down1, fuse1), dim=1))
        #
        #feat_sum = (attention4 + attention3+ attention2+ attention1)

        predict = F.interpolate(fuse1, scale_factor=self.output_stride//4,mode="bilinear",align_corners=True)
        # return predict4
        return predict

class MultiConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MultiConv, self).__init__()

        self.fuse_attn = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.PReLU()
            #nn.Softmax2d() if attn else
        )


    def forward(self, x):
        return self.fuse_attn(x)