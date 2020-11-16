import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.model_utils.aspp import build_aspp
from modeling.model_utils.decoder import build_decoder
from modeling.backbone import build_backbone



class DeepLab(nn.Module):
    def __init__(self, backbone,BatchNorm, output_stride, num_classes,freeze_bn=False):
        super(DeepLab, self).__init__()


        self.backbone=backbone
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)
        self.output_stride=output_stride
        if freeze_bn:
            self.freeze_bn()

    def forward(self, inputs):
        x = self.aspp(inputs[0])
        if(self.backbone=='xception'):#不同的backbone有不同的输出，处理不同
            low_level_feat=inputs[1]
        elif(self.backbone in ['resnet50','resnet101']):
            low_level_feat = inputs[3]
        else:
            NotImplementedError
        x = self.decoder(x, low_level_feat)
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



if __name__ == "__main__":
    model = DeepLab(backbone='resnet50', BatchNorm=nn.BatchNorm2d,output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 512, 512)
    output = model(input)
    print(output.size())


