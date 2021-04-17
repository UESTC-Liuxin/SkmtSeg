""" Full assembly of the parts to form the complete network """

from __future__ import division

import torch
import torch.nn as nn
from torch.nn.functional import upsample
from modeling.model_utils.non_local_parts import *
from modeling.model_utils.unet_parts import *
import torch.nn.functional as F
from modeling.model_utils.da_att import DANetHead

class NonlocalUNet(nn.Module):
    def __init__(self, backbone,BatchNorm, output_stride, num_classes,freeze_bn=False):
        super(NonlocalUNet, self).__init__()
        self.backbone = backbone
        self.n_classes = num_classes
        self.conv1 = nn.Conv2d(2048, 512, 1, bias=False)
        self.output_stride=output_stride
        if output_stride == 16:
            in_channels=2048
            inputstrides = [1024, 512, 128, 64]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError
        self.net = multi_head_attention_2d(2048, 2048, 2048, 512, 4, 0.5, 'SAME')

        self.head1 = DANetHead(64, 64, BatchNorm)
        self.head2 = DANetHead(256, 256, BatchNorm)
        self.head3 = DANetHead(512, 512, BatchNorm)
        self.head4 = DANetHead(2048, 512, BatchNorm)
        self.up1 = Up(1024, 512 //2,BatchNorm)
        self.up2 = Up(512, 256 // 4,BatchNorm)
        self.up3 = Up(128, 64,BatchNorm)
        self.outc = OutConv(64, num_classes)
        self.head= DANetHead(64, num_classes, BatchNorm)
        self.net_out = multi_head_attention_2d(64, 64, 64, num_classes, 4, 0.5, 'SAME')
        if freeze_bn:
            self.freeze_bn()

    def forward(self, x):
        #x2 = self.head1(x[5])
        #x3 = self.head2(x[3])
        #x4 = self.head3(x[2])
        x2 = x[5]
        x3 = x[3]
        x4 = x[2]
        #x5 = self.conv1(x[0])
        #x5 = self.head4(x[0])
        x5 = self.net(x[0])
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        logits = self.outc(x)
        #logits = self.net_out(x)
        x = F.interpolate(logits, scale_factor=2, mode='bilinear', align_corners=True)
        return x
