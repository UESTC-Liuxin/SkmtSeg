# -*- coding: utf-8 -*-
"""
@description:

@author: LiuXin
@contact: xinliu1996@163.com
@Created on: 2020/10/27 下午8:00
"""
import torch
import torch.hub
import torch, numpy as np
import torch.nn as nn
from modeling.backbone.resnet import ResNet,ResNet101
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.backbone import build_backbone



class FCN(torch.nn.Module):

    def __init__(self, backbone:str,BatchNorm,in_channels=[512,512,512,512],pretrained=True,upsample_ratio=8,n_class=21):

        super(FCN, self).__init__()

        backbone_model=build_backbone(backbone=backbone,BatchNorm=BatchNorm,output_stride=16)

        if(backbone=='mobilenet'):
            in_channels=[320,32,24,16]
        if(upsample_ratio==8):
            self.model=FCN8(backbone=backbone_model,BatchNorm=BatchNorm,in_channels=in_channels,num_classes=n_class)
        else:
            raise NotImplementedError

    def forward(self, x, debug=False):
        return self.model(x)

    def resume(self, file, test=False):
        pass
        # import torch
        # if test and not file:
        #     self.fcn = fcn_resnet50(pretrained=True, num_classes=21)
        #     return
        # if file:
        #     print('Loading checkpoint from: ' + file)
        #     checkpoint = torch.load(file)
        #     checkpoint = checkpoint['model_state_dict']
        #     self.load_state_dict(checkpoint)






def bilinear_init(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)



class FCN8(nn.Module):

    def __init__(self, backbone, BatchNorm,in_channels, num_classes=21):
        super(FCN8, self).__init__()
        self.backbone = backbone
        self.cls_num = num_classes

        self.relu = nn.ReLU(inplace=True)
        self.Conv1x1_x2 = nn.Conv2d(in_channels[0], self.cls_num, kernel_size=1)
        self.Conv1x1_x4 = nn.Conv2d(int(in_channels[1] ), self.cls_num, kernel_size=1)
        self.Conv1x1_x8 = nn.Conv2d(int(in_channels[2]), self.cls_num, kernel_size=1)
        self.Conv1x1_x16 = nn.Conv2d(int(in_channels[3]), self.cls_num, kernel_size=1)

        self.bn1 = BatchNorm(self.cls_num)
        self.DCN2 = nn.ConvTranspose2d(self.cls_num, self.cls_num, kernel_size=4, stride=2, dilation=1, padding=1)
        self.DCN2.weight.data = bilinear_init(self.cls_num, self.cls_num, 4)
        self.dbn2 = BatchNorm(self.cls_num)

        self.DCN4 = nn.ConvTranspose2d(self.cls_num, self.cls_num, kernel_size=4, stride=2, dilation=1, padding=1)
        self.DCN4.weight.data = bilinear_init(self.cls_num, self.cls_num, 4)
        self.dbn4 = BatchNorm(self.cls_num)

        self.DCN8 = nn.ConvTranspose2d(self.cls_num, self.cls_num, kernel_size=16, stride=8, dilation=1, padding=4)
        self.DCN8.weight.data = bilinear_init(self.cls_num, self.cls_num, 16)
        self.dbn8 = BatchNorm(self.cls_num)

    def forward(self, x):
        outputs = self.backbone(x)
        x = self.bn1(self.relu(self.Conv1x1_x2(outputs[0])))
        x5 = self.bn1(self.relu(self.Conv1x1_x4(outputs[1])))
        x = self.dbn2(self.relu(self.DCN2(x)))
        x = x + x5
        #
        x4 = self.bn1(self.relu(self.Conv1x1_x8(outputs[2])))
        x = self.dbn4(self.relu(self.DCN4(x)))
        x = x + x4
        if(x.size()[-1]!=512):
            x3 = self.bn1(self.relu(self.Conv1x1_x16(outputs[3])))
            x = self.dbn4(self.relu(self.DCN4(x)))
            x=x+x3
            x = self.dbn4(self.relu(self.DCN4(x)))
        else:
            x = self.dbn8(self.relu(self.DCN8(x)))

        return x


if __name__ == "__main__":
    from torchvision import transforms, utils

    device = 'cuda:0'
    model=FCN(backbone='mobilenet',sync_bn=True)
    tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    in_img = np.zeros((512, 512, 3), np.uint8)
    t_img = transforms.ToTensor()(in_img)
    t_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(t_img)
    t_img.unsqueeze_(0)
    t_img = t_img.to(device)
    model=model.to(device)
    x = model(t_img)
    print(x.shape)

