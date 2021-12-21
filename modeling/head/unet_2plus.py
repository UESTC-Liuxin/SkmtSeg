""" Full assembly of the parts to form the complete network """

from __future__ import division

import torch
import torch.nn as nn
from torch.nn.functional import upsample
from modeling.model_utils.unet_parts import *
import torch.nn.functional as F

class  UNet_Nested(nn.Module):
    def __init__(self, backbone,BatchNorm, output_stride, n_classes,freeze_bn=False):
        super(UNet_Nested, self).__init__()
        self.backbone = backbone
        self.conv1 = nn.Conv2d(2048, 512, 1, bias=False)
        self.output_stride=output_stride
        if output_stride == 16:
            filters = [32,64, 256, 512, 1024]

        elif output_stride == 8:
            filters = [32,64, 256, 512, 2048]
        else:
            raise NotImplementedError

        self.is_deconv = False
        self.is_batchnorm = True
        self.is_ds = True

        self.conv00 = unetConv2(3, filters[0], self.is_batchnorm)

        self.up_concat01 = unetUp(filters[1], filters[0], self.is_deconv)
        self.up_concat11 = unetUp(filters[2], filters[1], self.is_deconv, n_concat=2, n_in=0)
        self.up_concat21 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat31 = unetUp(filters[4], filters[3], self.is_deconv, n_concat=2,n_in=0)

        self.up_concat02 = unetUp(filters[1], filters[0], self.is_deconv,1,1)
        self.up_concat12 = unetUp(filters[2], filters[1], self.is_deconv, 3,0)
        self.up_concat22 = unetUp(filters[3], filters[2], self.is_deconv, 1)

        self.up_concat03 = unetUp(filters[1], filters[0], self.is_deconv, 2,1)
        self.up_concat13 = unetUp(filters[2], filters[1], self.is_deconv)

        self.up_concat04 = unetUp(filters[1], filters[0], self.is_deconv, 3,1)
        # final conv (without any concat)
        self.final_1 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_2 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_3 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_4 = nn.Conv2d(filters[0], n_classes, 1)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, x):
        X_00 = self.conv00(x[6])
        X_10 = x[5]             # 64*256*256
        X_20 = x[3]             # 256*128*128
        X_30 = x[2]             # 512*64*64
        X_40 = x[1]             # 2048*32*32

        X_01 = self.up_concat01(X_10, X_00)
        X_11 = self.up_concat11(X_20, X_10)
        X_21 = self.up_concat21(X_30, X_20)
        X_31 = self.up_concat31(X_40, X_30)
        # column : 2
        X_02 = self.up_concat02(X_11,  X_01, X_00)
        X_12 = self.up_concat12(X_21, X_10, X_11)
        X_22 = self.up_concat22(X_31, X_20, X_21)
        # column : 3
        X_03 = self.up_concat03(X_12, X_01, X_02,X_00)
        X_13 = self.up_concat13(X_22, X_10, X_11, X_12)
        # column : 4
        X_04= self.up_concat04(X_13, X_01, X_02, X_03,X_00)

        final_1 = self.final_1(X_01)
        final_2 = self.final_2(X_02)
        final_3 = self.final_3(X_03)
        final_4 = self.final_4(X_04)

        final = (final_1 + final_2 + final_3 + final_4) / 4
        if self.is_ds:
            return final
        else:
            return [final_1,final_2,final_3,final_4]

