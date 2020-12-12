###########################################################################
# Created by: CASIA IVA 
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################
from __future__ import division
import torch.nn as nn
import torch
import torch.nn.functional as F
from modeling.model_utils.da_att import PAM_Module
from modeling.model_utils.da_att import CAM_Module

class MultiScaleAttention(nn.Module):
    def __init__(self, backbone,BatchNorm, output_stride, num_classes,freeze_bn=False):
        super(MultiScaleAttention, self).__init__()
        self.backbone = backbone
        if(backbone in ["resnet50","resnet101"]):
            in_channels=2048
        else:
            raise NotImplementedError
        # self.down4 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU())
        # self.down3 = nn.Sequential(nn.Conv2d(in_channels//2, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU())
        # self.down2 = nn.Sequential(nn.Conv2d(in_channels//4, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU())
        self.down1 = nn.Sequential(nn.Conv2d(in_channels//8, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU())

        self.fuse1 = MultiConv(256, 64)

        self.head1 = DANetHead(in_channels, 64, BatchNorm)
        self.head2 = DANetHead(in_channels//2, 64, BatchNorm)
        self.head3 = DANetHead(in_channels//4, 64, BatchNorm)
        self.head4 = DANetHead(in_channels//8, 64, BatchNorm)

        self.output_stride = output_stride

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, 5, 1))
        if freeze_bn:
            self.freeze_bn()

    def forward(self, inputs):
        inputs1 = self.head1(inputs[0])
        inputs2 = self.head2(inputs[1])
        inputs3 = self.head3(inputs[2])
        #inputs4 = inputs[3]
        down4 = F.upsample(inputs1, size=inputs[3].size()[2:], mode='bilinear')
        down3 = F.upsample(inputs2, size=inputs[3].size()[2:], mode='bilinear')
        down2 = F.upsample(inputs3, size=inputs[3].size()[2:], mode='bilinear')
        down1 = self.down1(inputs[3])

        fuse1 = self.fuse1(torch.cat((down4, down3, down2, down1), 1))
        fuse1 = self.conv8(fuse1)
        #attention4 = self.head4(torch.cat((down4, fuse1), dim=1))
        # attention3 = self.head3(torch.cat((down4, fuse1), dim=1))
        # attention2 = self.head2(torch.cat((down4, fuse1), dim=1))
        # attention1 = self.head1(torch.cat((down4, fuse1), dim=1))
        #
        # feat_sum = attention4 + attention3+ attention2+ attention1

        predict = F.interpolate(fuse1, scale_factor=self.output_stride//4,mode="bilinear",align_corners=True)
        # return predict4
        return predict
        
class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   BatchNorm(inter_channels),
                                   nn.ReLU())
        
        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   BatchNorm(inter_channels),
                                   nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   BatchNorm(inter_channels),
                                   nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   BatchNorm(inter_channels),
                                   nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv+sc_conv
        
        sasc_output = self.conv8(feat_sum)

        '''output = [sasc_output]
        output.append(sa_output)
        output.append(sc_output)'''
        return sasc_output


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