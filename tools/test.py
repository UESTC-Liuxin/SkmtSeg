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
from metrics.metrics import Evaluator
from tqdm import tqdm
from prettytable import PrettyTable


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

        # Define Evaluator
        self.evaluator = Evaluator(args.num_classes)

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
        self.evaluator.reset()

        tloss = []
        with torch.no_grad():
            pbar = tqdm(self.dataloader, ncols=100)
            for iter, batch in enumerate(pbar):
                pbar.set_description("Testing Processing epoach:{}".format(epoch))
                start_time = time.time()

                batch = self.dict_to_cuda(batch)
                if self.args.deep_supervision:
                    outputs = self.model(batch)
                    loss = 0
                    for outputl in outputs['trunk_out']:
                        sample = {'trunk_out': outputl, 'auxiliary_out': outputs['auxiliary_out']}
                        loss += self.criterion(sample, batch['label']).cuda()
                    loss /= len(outputs)
                    output = {'trunk_out': outputs['trunk_out'][-1], 'auxiliary_out': outputs['auxiliary_out']}
                else:
                    output = self.model(batch)
                    loss = self.criterion(output, batch['label']).cuda()
                tloss.append(loss.item())
                pred = np.asarray(np.argmax(output['trunk_out'][0].cpu().detach(), axis=0), dtype=np.uint8)
                gt=np.asarray(batch['label'].cpu().detach().squeeze(0), dtype=np.uint8)


                self.visualize(gt, pred, iter, writer,"test")
                self.evaluator.add_batch(gt, pred)


        self.logger.info('======>epoch:{}---loss:{:.3f}'.format(epoch,sum(tloss)/len(tloss)))
        writer.add_scalar('test/loss_epoch', sum(tloss)/len(tloss), epoch)

        #add a tabel
        tb_overall = PrettyTable()
        tb_cls  =PrettyTable()
        # Fast test during the training
        Acc = np.around(self.evaluator.Pixel_Accuracy(),decimals=3)
        mAcc = np.around(self.evaluator.Pixel_Accuracy_Class(),decimals=3)
        mIoU = np.around(self.evaluator.Mean_Intersection_over_Union(),decimals=3)
        FWIoU = np.around(self.evaluator.Frequency_Weighted_Intersection_over_Union(),decimals=3)
        acc_cls = np.around(self.evaluator.Acc_Class(),decimals=3)
        iou_cls =  np.around(self.evaluator.IoU_Class(),decimals=3)
        confusion_matrix=  np.around(self.evaluator.confusion_matrix,decimals=3)
        #Print info
        tb_overall.field_names = ["Acc", "mAcc", "mIoU", "FWIoU"]
        tb_overall.add_row([Acc, mAcc, mIoU, FWIoU])

        tb_cls.field_names =['Index']+list(self.dataloader.dataset.CLASSES[:self.args.num_classes])
        tb_cls.add_row(['acc']+list(acc_cls))
        tb_cls.add_row(['iou'] + list(iou_cls))
        self.logger.info(tb_overall)
        self.logger.info(tb_cls)


        return Acc,mAcc,mIoU,FWIoU,confusion_matrix





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










