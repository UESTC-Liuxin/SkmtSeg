# -*- coding: utf-8 -*-
"""
@description:

@author: LiuXin
@contact: xinliu1996@163.com
@Created on: 2020/11/2 上午11:12
"""

# -*- coding: utf-8 -*-

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from metrics.metrics import runningScore, averageMeter
from tqdm import tqdm



class Tester(object):

    def __init__(self,args,dataloader:DataLoader,model:nn.Module,
                 criterion,logger,summary=None):
        """

        :param args:
        :param dataloader:
        :param model:
        :param optimizer:
        :param criterion:
        :param logger:
        :param summary:
        """
        self.args=args
        self.dataloader=dataloader
        self.model = model
        self.logger=logger
        self.summary=summary
        self.criterion=criterion

        self.running_Metrics = runningScore(args.num_classes)

    def dict_to_cuda(self,tensors):
        cuda_tensors={}
        for key,value in tensors.items():
            if(isinstance(value,torch.Tensor)):
                value=value.cuda()
            cuda_tensors[key]=value
        return cuda_tensors

    def dict_to_cpu(self,tensors):
        cuda_tensors={}
        for key,value in tensors.items():
            if(isinstance(value,torch.Tensor)):
                value=value.cpu()
            cuda_tensors[key]=value
        return cuda_tensors

    def test_one_epoch(self,epoch,writer):
        """

        :param epoch:
        :return:
        """
        self.model.eval()
        total_batches = len(self.dataloader)

        tloss = []
        with torch.no_grad():
            pbar = tqdm(self.dataloader, ncols=100)
            for iter, batch in enumerate(pbar):
                pbar.set_description("Testing Processing epoach:{}".format(epoch))
                start_time = time.time()

                batch = self.dict_to_cuda(batch)
                output = self.model(batch)
                loss = self.criterion(output, batch['label']).cuda()

                tloss.append(loss.item())

                gt=np.asarray(batch['label'].cpu().detach().squeeze(0), dtype=np.uint8)
                pred = np.asarray(np.argmax(output['trunk_out'][0].cpu().detach(), axis=0), dtype=np.uint8)

                self.running_Metrics.update(gt, pred)
                self.visualize(gt, pred, iter, writer,"test")

        self.logger.info('======>epoch:{}---loss:{:.3f}'.format(epoch,sum(tloss)/len(tloss)))
        writer.add_scalar('test/loss_epoch', sum(tloss)/len(tloss), epoch)
        score, class_iou, class_acc,class_F1 = self.running_Metrics.get_scores()
        self.running_Metrics.reset()



        return score, class_iou,class_acc, class_F1



    def visualize(self,gt,pred,epoch,writer,title):
        """

        :param input:
        :param output:
        :param index:
        :return:
        """
        gt = self.dataloader.dataset.decode_segmap(gt)

        pred=self.dataloader.dataset.decode_segmap(pred)
        self.summary.visualize_image(writer,title+'/gt',gt,epoch)
        self.summary.visualize_image(writer, title+'/pred', pred, epoch)

#TODO:用于调试的visualize代码，观察取的图片和裁剪的图片是否有问题
def visualize(img,tag):
    import matplotlib.pyplot as plt  # plt 用于显示图片
    import matplotlib.image as mpimg  # mpimg 用于读取图片
    import torchvision.transforms as transforms
    import numpy as np

    if(isinstance(img,torch.Tensor)):
        unloader = transforms.ToPILImage()
        img = img.cpu().clone().detach()
        img=unloader(img)
        img=np.array(img)

    plt.title(tag)
    plt.imshow(img)
    plt.show()










