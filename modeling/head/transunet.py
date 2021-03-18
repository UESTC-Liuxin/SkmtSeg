""" Full assembly of the parts to form the complete network """

from __future__ import division

import torch
import torch.nn as nn
from modeling.model_utils.unet_parts import *
from modeling.model_utils.transformer import *
from utils.utils import init_weights,count_param

class TransUNet(nn.Module):
    def __init__(self, backbone, BatchNorm, output_stride, num_classes,img_size, freeze_bn=False):
        super(TransUNet, self).__init__()
        self.backbone = backbone
        self.n_classes = num_classes
        self.conv1 = nn.Conv2d(2048, 512, 1, bias=False)
        self.output_stride = output_stride
        self.transformer = Transformer(img_size, patch_size=2, in_channels=512,out_channels=512,num_heads = 16, hidden_size=512,num_layers=8, vis=False)

        self.up1 = Up(1024, 512 // 2, BatchNorm)
        self.up2 = Up(512, 256 // 4, BatchNorm)
        self.up3 = Up(128, 64, BatchNorm)
        self.outc = OutConv(64, num_classes)
        if freeze_bn:
            self.freeze_bn()

    def forward(self, x):
        x2 = x[5]
        #print("x1", x[4].shape)#
        #print("x2", x2.shape)#n*64*256*256
        x3 = x[3]
        #print("x3", x3.shape)#n*256*128*128
        x4 = x[2]
        #print("x4:", x4.shape)#n*512*64*64
        x5 = self.conv1(x[0])
        x5 = self.transformer(x5)#n*512*2*2
        #print("x5:", x5.shape)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        logits = self.outc(x)
        x = F.interpolate(logits, scale_factor=2, mode='bilinear', align_corners=True)
        return x