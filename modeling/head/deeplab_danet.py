import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.model_utils.aspp import build_aspp
from modeling.model_utils.decoder import build_decoder
<<<<<<< HEAD
from modeling.backbone import build_backbone
from modeling.model_utils.da_att import PAM_Module
from modeling.model_utils.da_att import CAM_Module
from modeling.model_utils.backbone2head import get_inchannels,get_low_level_feat
from modeling.head.danet import DANetHead
=======
from modeling.model_utils.da_att import DANetHead
>>>>>>> origin/danet

from modeling.model_utils.non_local_parts import *

class DeepLabDANet(nn.Module):
    def __init__(self, backbone,BatchNorm, output_stride,num_classes,freeze_bn=False):
        super(DeepLabDANet, self).__init__()
        self.backbone=backbone
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)
        self.backbone = backbone
<<<<<<< HEAD
        in_channels=get_inchannels(self.backbone)[0]
=======
        if (backbone in ["resnet50", "resnet101"]):
            in_channels = 2048
        else:
            raise NotImplementedError
        #self.net = multi_head_attention_2d(in_channels, in_channels, in_channels, num_classes, 4, 0.5, 'SAME')
>>>>>>> origin/danet
        self.head = DANetHead(in_channels, num_classes, BatchNorm)
        self.output_stride = output_stride
        if freeze_bn:
            self.freeze_bn()

    def forward(self, inputs):
        x0 = self.head(inputs[0])
        x = self.aspp(inputs[0])
        low_level_feat=get_low_level_feat(self.backbone)

        x = self.decoder(x, low_level_feat)
        x0 = F.interpolate(x0, scale_factor=4, mode='bilinear', align_corners=True)
        x = x0+x
        x = F.interpolate(x, scale_factor=self.output_stride/4, mode='bilinear', align_corners=True)
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


