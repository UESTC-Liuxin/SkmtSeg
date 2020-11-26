# -*- coding: utf-8 -*-
"""
@description:

@author: LiuXin
@contact: xinliu1996@163.com
@Created on: 2020/11/19 上午11:28
"""
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from tqdm import tqdm
import PIL.Image as Image
from modeling import build_skmtnet
import os


class Inferencer(object):

    def __init__(self, args, model: nn.Module):
        """

        :param args:
        :param model:
        """
        self.args = args
        self.model = model
        self.model.eval()

    def dict_to_cuda(self, tensors):
        cuda_tensors = {}
        for key, value in tensors.items():
            if (isinstance(value, torch.Tensor)):
                value = value.cuda()
            cuda_tensors[key] = value
        return cuda_tensors

    def dict_to_cpu(self, tensors):
        cuda_tensors = {}
        for key, value in tensors.items():
            if (isinstance(value, torch.Tensor)):
                value = value.cpu()
            cuda_tensors[key] = value
        return cuda_tensors

    def inference(self, img):
        """

        :param epoch:
        :return:
        """
        with torch.no_grad():
            img = self.dict_to_cuda(img)
            output = self.model(img)
            pred = np.asarray(np.argmax(output['out'].squeeze(0).cpu().detach(), axis=0), dtype=np.uint8)

    def save(self, mask, name):
        """

        :param mask:
        :param name:
        :return:
        """
        pred = self.dataloader.dataset.decode_segmap(mask)
        img = Image.fromarray(pred)
        img.save(os.path.join("results", name))

    def visualize(self, gt, pred, epoch, writer, title):
        """

        :param input:
        :param output:
        :param index:
        :return:
        """
        gt = self.dataloader.dataset.decode_segmap(gt)

        pred = self.dataloader.dataset.decode_segmap(pred)


def SegSkmt(args):
    # build model
    model = build_skmtnet(backbone='resnet50', auxiliary_head=args.auxiliary, trunk_head=args.trunk_head,
                          num_classes=args.num_classes, output_stride=16)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    checkpoint = torch.load(args.model)
    print("loading model...........")
    model = model.load_state_dict(checkpoint)
    infer = Inferencer(args, model)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    start_time = time.time()
    if (os.path.isfile(args.imgs_path)):
        img = Image.open(args.imgs_path)
        img = transform(img).unsquzee(0).to(device)
        infer.inference(img)
    else:
        files = os.listdir(args.imgs_path)
        for i, img_name in enumerate(tqdm(files)):
            img = Image.open(img_name)
            img = transform(img).unsquzee(0).to(device)
            infer.inference(img)
    end_time = time.time()
    cost_time = end_time - start_time
    print("finish it,cost ：%.8s s" % cost_time)


def uzip_model(args):
    # 在torch 1.6版本中重新加载一下网络参数

    model = build_skmtnet(backbone='resnet50', auxiliary_head=args.auxiliary, trunk_head=args.trunk_head,
                          num_classes=args.num_classes, output_stride=16)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model)
    model.to(device)
    model.load_state_dict(torch.load(args.model))  # 加载模型参数，model_cp为之前训练好的模型参数（zip格式）
    # 重新保存网络参数，此时注意改为非zip格式
    torch.save(model.state_dict(), args.model, _use_new_zipfile_serialization=False)


if __name__ == "__main__":
    import timeit

    start = timeit.default_timer()

    parser = argparse.ArgumentParser(description='Semantic Segmentation...')

    parser.add_argument('--model', default='checkpoints/model.pth', type=str)
    parser.add_argument('--imgs_path', default='data/SKMT/Seg/JPEGImages', type=str)
    parser.add_argument('--num_classes', default=19, type=int)
    parser.add_argument('--auxiliary', default=None, type=str)
    parser.add_argument('--trunk_head', default='deeplab', type=str)
    parser.add_argument('--savedir', default="./results", help="directory to save the model snapshot")
    parser.add_argument('--gpus', type=str, default='0')

    args = parser.parse_args()

    SegSkmt(args, )
