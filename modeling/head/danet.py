###########################################################################
# Created by: CASIA IVA 
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################
from __future__ import division
import torch.nn as nn
import torch.nn.functional as F
from modeling.model_utils.da_att import DANetHead

class DANet(nn.Module):
    def __init__(self, backbone,BatchNorm, output_stride, num_classes,freeze_bn=False):
        super(DANet, self).__init__()
        self.backbone = backbone
        if(backbone in ["resnet50","resnet101"]):
            in_channels=2048
        else:
            raise NotImplementedError

        self.head = DANetHead(in_channels, num_classes, BatchNorm)
        self.output_stride = output_stride*2
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
