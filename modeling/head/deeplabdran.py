###########################################################################
# Created by: CASIA IVA  
# Email: jliu@nlpr.ia.ac.cn 
# Copyright (c) 2020
###########################################################################
from __future__ import division

import torch
import torch.nn as nn
from torch.nn.functional import upsample
from modeling.model_utils.aspp import build_aspp
from modeling.model_utils.decoder import build_decoder
from modeling.model_utils.dran_att import CPAMDec,CCAMDec,CPAMEnc, CLGD
from modeling.head.dran import DranHead
import torch.nn.functional as F

up_kwargs = {'mode': 'bilinear', 'align_corners': True}


class DeepDran(nn.Module):
    def __init__(self, backbone,BatchNorm, output_stride, num_classes,freeze_bn=False):
        super(DeepDran, self).__init__( )
        self.backbone = backbone
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)
        if (backbone in ["resnet50", "resnet101"]):
            in_channels = 2048
            in_channels_seg = 256
        else:
            raise NotImplementedError

        self.head = DranHead(in_channels, num_classes, BatchNorm)
        self.cls_seg = nn.Sequential(nn.Dropout2d(0.1, False),
                   nn.Conv2d(in_channels_seg, num_classes, 1))
        self.output_stride = output_stride
        if freeze_bn:
            self.freeze_bn()

    def forward(self, inputs):
        x = self.aspp(inputs[0])
        if (self.backbone == 'xception'):  # 不同的backbone有不同的输出，处理不同
            low_level_feat = inputs[1]
        elif (self.backbone in ['resnet50', 'resnet101']):
            low_level_feat = inputs[3]
        else:
            NotImplementedError
        x = self.decoder(x, low_level_feat)
        #x = F.interpolate(x, scale_factor=self.output_stride / 4, mode='bilinear', align_corners=True)
        ## dran head for seg

        final_feat = self.head(inputs)
        cls_seg = self.cls_seg(final_feat)
        x = cls_seg + x
        x = F.interpolate(x, scale_factor=self.output_stride/4, mode="bilinear", align_corners=True)
        return x

class DranHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(DranHead, self).__init__()
        inter_channels = in_channels // 4

        ## Convs or modules for CPAM 
        self.conv_cpam_b = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU()) # conv5_s
        self.cpam_enc = CPAMEnc(inter_channels, norm_layer) # en_s
        self.cpam_dec = CPAMDec(inter_channels) # de_s
        self.conv_cpam_e = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                           norm_layer(inter_channels),
                           nn.ReLU()) # conv52

        ## Convs or modules for CCAM
        self.conv_ccam_b = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU()) # conv5_c
        self.ccam_enc = nn.Sequential(nn.Conv2d(inter_channels, inter_channels//16, 1, bias=False),
                                   norm_layer(inter_channels//16),
                                   nn.ReLU()) # conv51_c
        self.ccam_dec = CCAMDec() # de_c
        self.conv_ccam_e = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU()) # conv51

        ## Fusion conv
        self.conv_cat = nn.Sequential(nn.Conv2d(inter_channels*2, inter_channels//2, 3, padding=1, bias=False),
                                   norm_layer(inter_channels//2),
                                   nn.ReLU()) # conv_f
        ## Cross-level Gating Decoder(CLGD) 
        self.clgd = CLGD(inter_channels//2,inter_channels//2,norm_layer)

    def forward(self, multix):

        ## Compact Channel Attention Module(CCAM)
        ccam_b = self.conv_ccam_b(multix[0])
        ccam_f = self.ccam_enc(ccam_b)
        ccam_feat = self.ccam_dec(ccam_b,ccam_f)        
        
        ## Compact Spatial Attention Module(CPAM)
        cpam_b = self.conv_cpam_b(multix[0])
        cpam_f = self.cpam_enc(cpam_b).permute(0,2,1)#BKD
        cpam_feat = self.cpam_dec(cpam_b,cpam_f)

        ## Fuse two modules
        ccam_feat = self.conv_ccam_e(ccam_feat)
        cpam_feat = self.conv_cpam_e(cpam_feat)
        feat_sum = self.conv_cat(torch.cat([cpam_feat,ccam_feat],1))
        
        ## Cross-level Gating Decoder(CLGD) 
        final_feat = self.clgd(multix[3], feat_sum)

        return final_feat

