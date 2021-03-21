""" Full assembly of the parts to form the complete network """

from __future__ import division

import torch
import torch.nn as nn

from modeling.model_utils.da_att import DANetHead
from modeling.model_utils.bifusion import BiFusion,Attention_block
from modeling.model_utils.unet_parts import *
from modeling.model_utils.transformer import *
from utils.utils import init_weights,count_param

class TransFuse(nn.Module):
    def __init__(self, backbone, BatchNorm, output_stride, num_classes,img_size, freeze_bn=False):
        super(TransFuse, self).__init__()

        self.backbone = backbone
        self.n_classes = num_classes
        self.conv1 = nn.Conv2d(2048, 512, 1, bias=False)
        self.output_stride = output_stride
        self.fusion1 = BiFusion(512, 512, BatchNorm)
        self.fusion2 = BiFusion(512, 256, BatchNorm)
        self.fusion3 = BiFusion(256, 64, BatchNorm)
        self.fusion4 = BiFusion(64, 32, BatchNorm)
        self.avepool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.transformer = Transformer(img_size, patch_size=16, in_channels=3, out_channels=512,
                                       num_heads=16,hidden_size=256, num_layers=8, vis=False)

        self.conv2 = nn.Conv2d(256*3, 256, 1, bias=False)
        self.conv3 = nn.Conv2d(64*5, 64, 1, bias=False)
        self.conv4 = nn.Conv2d(32*3, 64, 1, bias=False)

        self.cov2= nn.Conv2d(512, 256, 1, bias=False)
        self.cov3 = nn.Conv2d(256, 64, 1, bias=False)

        self.ag2 = Attention_block(F_g=512,F_l=256,F_int=256)
        self.ag3 = Attention_block(F_g=256,F_l=64,F_int=64)
        self.ag4 = Attention_block(F_g=64,F_l=32,F_int=32)

        self.up1 = Up(1024, 512 //2, BatchNorm)
        self.up2 = Up(512, 256 // 4, BatchNorm)
        self.up3 = Up(128, 64, BatchNorm)
        self.outc = OutConv(64, self.n_classes)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, x):
        x_tran = self.transformer(x[6])  # n*512*32*32
        x2 = x[5]#n*64*256*256
        x3 = x[3]#n*256*128*128
        x4 = x[2]#n*512*64*64
        x5 = self.conv1(x[0])#n*512*32*32
        f0 = self.fusion1(x5,x_tran)#512*32*32

        t1 = F.interpolate(x_tran, scale_factor=2, mode='bilinear', align_corners=True)#512*64*64
        f1 = self.fusion2(x4, t1)#256*64*64

        t2 = F.interpolate(t1, scale_factor=2, mode='bilinear', align_corners=True)
        t2 = self.cov2(t2)
        f2 = self.fusion3(x3, t2)#64*128*128

        t3 = F.interpolate(t2, scale_factor=2, mode='bilinear', align_corners=True)
        t3 = self.cov3(t3)
        f3 = self.fusion4(x2, t3)#64*256*256

        f0 = F.interpolate(f0, scale_factor=2, mode='bilinear', align_corners=True)
        f1 = self.ag2(g=f0, x=f1)
        f1 = self.conv2(torch.cat([f1, f0], dim=1))

        f1 = F.interpolate(f1, scale_factor=2, mode='bilinear', align_corners=True)
        f2 = self.ag3(g=f1,x=f2)
        f2 = self.conv3(torch.cat([f2, f1], dim=1))

        f2 = F.interpolate(f2, scale_factor=2, mode='bilinear', align_corners=True)
        f3 = self.ag4(g=f2, x=f3)
        f3 = self.conv4(torch.cat([f3, f2], dim=1))

        logits = self.outc(f3)
        x = F.interpolate(logits, scale_factor=2, mode='bilinear', align_corners=True)
        return x