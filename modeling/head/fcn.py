# -*- coding: utf-8 -*-
"""
@description:

@author: LiuXin
@contact: xinliu1996@163.com
@Created on: 2020/11/9 下午3:31
"""

# -*- coding: utf-8 -*-

import torch
import torch.hub
import torch, numpy as np
import torch.nn as nn


class FCN(torch.nn.Module):

    def __init__(self, backbone:str,BatchNorm,output_stride,num_classes):

        super(FCN, self).__init__()

        self.BatchNorm=BatchNorm

        if(backbone=='mobilenet'):
            in_channels=[320,32,24,16]
        elif(backbone=="resnet50"):
            in_channels=[2048,1024,512,256]
        else:
            raise ValueError

        if(output_stride==8):
            self.model=FCN8(BatchNorm=BatchNorm,in_channels=in_channels,num_classes=num_classes)
        elif(output_stride==16):
            self.model=FCN16(BatchNorm=BatchNorm,in_channels=in_channels,num_classes=num_classes)
        else:
            raise NotImplementedError

    def forward(self, inputs):
        return self.model(inputs)


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, self.BatchNorm):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()


    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], self.BatchNorm) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p





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

    def __init__(self, BatchNorm,in_channels, num_classes=21):
        super(FCN8, self).__init__()
        self.num_classes = num_classes

        self.relu = nn.ReLU(inplace=True)
        self.Conv1x1_list=nn.Sequential()
        self.in_channels=in_channels
        for in_channel in in_channels:
            self.Conv1x1_list.add_module(str(in_channel),nn.Conv2d(in_channel, self.num_classes, kernel_size=1))
        self.bn1 = BatchNorm(self.num_classes)

        self.DCN2 = nn.ConvTranspose2d(self.num_classes, self.num_classes, kernel_size=4, stride=2, dilation=1, padding=1)
        self.DCN2.weight.data = bilinear_init(self.num_classes, self.num_classes, 4)
        self.dbn2 = BatchNorm(self.num_classes)

        self.DCN4 = nn.ConvTranspose2d(self.num_classes, self.num_classes, kernel_size=8, stride=4, dilation=1, padding=2)
        self.DCN4.weight.data = bilinear_init(self.num_classes, self.num_classes, 8)
        self.dbn4 = BatchNorm(self.num_classes)

    def forward(self, inputs:[tuple,list]):

        x5 = self.bn1(self.relu(self.Conv1x1_list[0](inputs[0])))
        x3 = self.bn1(self.relu(self.Conv1x1_list[2](inputs[2])))
        #相加，上采样X2
        x = x5 + x3
        x = self.dbn2(self.relu(self.DCN2(x)))
        #相加，上采样X2
        x2 = self.bn1(self.relu(self.Conv1x1_list[3](inputs[3])))
        x=x+x2
        x = self.dbn4(self.relu(self.DCN4(x)))

        return x



class FCN16(nn.Module):

    def __init__(self, BatchNorm,in_channels, num_classes=21):
        super(FCN16, self).__init__()
        self.num_classes = num_classes

        self.relu = nn.ReLU(inplace=True)
        self.Conv1x1_list=nn.Sequential()
        self.in_channels=in_channels
        for in_channel in in_channels:
            self.Conv1x1_list.add_module(str(in_channel),nn.Conv2d(in_channel, self.num_classes, kernel_size=1))
        self.bn1 = BatchNorm(self.num_classes)

        self.bn1 = BatchNorm(self.num_classes)
        self.DCN2 = nn.ConvTranspose2d(self.num_classes, self.num_classes, kernel_size=4, stride=2, dilation=1, padding=1)
        self.DCN2.weight.data = bilinear_init(self.num_classes, self.num_classes, 4)
        self.dbn2 = BatchNorm(self.num_classes)

        self.DCN8 = nn.ConvTranspose2d(self.num_classes, self.num_classes, kernel_size=16, stride=8, dilation=1, padding=4)
        self.DCN8.weight.data = bilinear_init(self.num_classes, self.num_classes, 16)
        self.dbn8 = BatchNorm(self.num_classes)

    def forward(self, inputs:[tuple,list]):
        x5 = self.bn1(self.relu(self.Conv1x1_list[0](inputs[0])))
        x4 = self.bn1(self.relu(self.Conv1x1_list[1](inputs[1])))
        x=x5+x4
        x = self.dbn2(self.relu(self.DCN2(x)))
        x3 = self.bn1(self.relu(self.Conv1x1_list[2](inputs[2])))
        x = x + x3
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
