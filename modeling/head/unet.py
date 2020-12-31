""" Full assembly of the parts to form the complete network """

from __future__ import division

import torch
import torch.nn as nn
from modeling.model_utils.unet_parts import *

from utils.utils import init_weights,count_param

class UNet(nn.Module):
    def __init__(self, backbone,BatchNorm, output_stride, num_classes,freeze_bn=False):
        super(UNet, self).__init__()
        self.in_channels = 3
        self.feature_scale = 2
        self.is_deconv = True
        self.is_batchnorm = True

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)
        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], num_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)  # 16*512*512
        maxpool1 = self.maxpool(conv1)  # 16*256*256

        conv2 = self.conv2(maxpool1)  # 32*256*256
        maxpool2 = self.maxpool(conv2)  # 32*128*128

        conv3 = self.conv3(maxpool2)  # 64*128*128
        maxpool3 = self.maxpool(conv3)  # 64*64*64

        conv4 = self.conv4(maxpool3)  # 128*64*64
        maxpool4 = self.maxpool(conv4)  # 128*32*32

        center = self.center(maxpool4)  # 256*32*32
        up4 = self.up_concat4(center, conv4)  # 128*64*64
        up3 = self.up_concat3(up4, conv3)  # 64*128*128
        up2 = self.up_concat2(up3, conv2)  # 32*256*256
        up1 = self.up_concat1(up2, conv1)  # 16*512*512

        final = self.final(up1)

        return final