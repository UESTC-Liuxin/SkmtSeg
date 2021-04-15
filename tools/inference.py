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
import skimage
import torch
import torch.nn as nn
import torchvision.transforms as T
from tqdm import tqdm
import PIL.Image as Image
from modeling import build_skmtnet
import os
from dataloader.skmt import SkmtDataSet
from tools.postprocess import postprocess


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
            pred = np.asarray(np.argmax(output['trunk_out'].squeeze(0).cpu().detach(), axis=0), dtype=np.uint8)
        return pred

    def save(self, mask, name):
        """

        :param mask:
        :param name:
        :return:
        """
        pred = decode_segmap(mask)
        pred = Image.fromarray(skimage.util.img_as_ubyte(pred))
        pred.save(os.path.join(args.savedir, name))

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
                          num_classes=args.num_classes, output_stride=16, img_size=args.crop_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    checkpoint = torch.load(args.model)
    print("loading model...........")
    model.load_state_dict(checkpoint["model"])
    infer = Inferencer(args, model)

    transform = T.Compose([
        T.Resize([args.crop_size,args.crop_size]),
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
            img = Image.open(os.path.join(args.imgs_path, img_name)).convert('RGB')
            img = transform(img)

            # img = np.array(img).astype(np.float32).transpose((2, 0, 1))
            # img = torch.from_numpy(img).float()

            img = torch.unsqueeze(img, dim=0)
            img = img.to(device)
            sample = {'image': img}
            pre = infer.inference(sample)
            post = postprocess(pre, args.num_classes)
            infer.save(post,img_name)
    end_time = time.time()
    cost_time = end_time - start_time
    print("finish it,cost ：%.8s s" % cost_time)


def uzip_model(args):
    # 在torch 1.6版本中重新加载一下网络参数
    model = build_skmtnet(backbone='resnet50', auxiliary_head=args.auxiliary, trunk_head=args.trunk_head,
                          num_classes=args.num_classes, output_stride=16, img_size=args.crop_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model)
    model.to(device)
    model.load_state_dict(torch.load(args.model))  # 加载模型参数，model_cp为之前训练好的模型参数（zip格式）
    # 重新保存网络参数，此时注意改为非zip格式
    torch.save(model.state_dict(), args.model, _use_new_zipfile_serialization=False)
def decode_segmap(label_mask):
    label_colours = SkmtDataSet.PALETTE
    n_classes = len(label_colours)

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb

if __name__ == "__main__":
    import timeit
    start = timeit.default_timer()
    parser = argparse.ArgumentParser(description='Semantic Segmentation...')
    parser.add_argument('--model', default='../checkpoints/best_model.pth', type=str)
    parser.add_argument('--imgs_path', default='../data/SKMT/Seg/JPEGImages', type=str)
    parser.add_argument('--crop_size', default=512, type=int)
    parser.add_argument('--num_classes', default=11, type=int)
    parser.add_argument('--auxiliary', default='fcn', type=str)
    parser.add_argument('--trunk_head', default='deeplab_danet', type=str)
    parser.add_argument('--savedir', default="results", help="directory to save the model snapshot")
    parser.add_argument('--gpus', type=str, default='0')
    args = parser.parse_args()

    SegSkmt(args, )
