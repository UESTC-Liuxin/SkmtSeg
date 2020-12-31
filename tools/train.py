# -*- coding: utf-8 -*-
"""
@description:

@author: LiuXin
@contact: xinliu1996@163.com
@Created on: 2020/11/1 下午10:22
"""
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.lr_scheduler import LR_Scheduler

class Trainer(object):

    def __init__(self,args,dataloader:DataLoader,model:nn.Module,optimizer,
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
        self.optimizer=optimizer
        self.start_epoch=0
        # Define lr scheduler
        self.scheduler = LR_Scheduler('poly', args.lr,args.max_epochs, len(self.dataloader))
        #进行训练恢复
        if(args.resume):
            self.resume()

    def resume(self):
        self.logger.info("---------------resume beginning....")
        checkpoint=torch.load(self.args.resume)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.criterion.load_state_dict(checkpoint['criterion'])
        self.start_epoch=checkpoint['epoch']
        self.logger.info("---------------resume end....")



    def dict_to_cuda(self,tensors):
        cuda_tensors={}
        for key,value in tensors.items():
            if(isinstance(value,torch.Tensor)):
                value=value.cuda()
            cuda_tensors[key]=value
        return cuda_tensors

    def train_one_epoch(self,epoch,writer,best_pred):
        """

        :param epoch:
        :return:
        """
        self.model.train()
        total_batches = len(self.dataloader)
        tloss = []
        pbar=tqdm(self.dataloader,ncols=100)
        for iter, batch in enumerate(pbar):
            pbar.set_description("Training Processing epoach:{}".format(epoch))
            self.scheduler(self.optimizer, iter, epoch, best_pred)

            # start_time = time.time()
            batch=self.dict_to_cuda(batch)
            output=self.model(batch)

            loss = self.criterion(output,batch['label'])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            tloss.append(loss.item())

            if (iter % self.args.show_interval == 0):
                pred=np.asarray(np.argmax(output['trunk_out'][0].cpu().detach(), axis=0), dtype=np.uint8)
                gt = batch['label'][0]  #每次显示第一张图片
                gt=np.asarray(gt.cpu(), dtype=np.uint8)
                self.visualize(gt, pred, epoch*1000+iter,writer,"train")


        self.logger.info('======>epoch:{}---loss:{:.3f}'.format(epoch,sum(tloss)/len(tloss)))
        writer.add_scalar('train/loss_epoch', sum(tloss)/len(tloss), epoch)

    def visualize(self,gt,pred,epoch,writer,title):
        """

        :param input:
        :param output:
        :param index:
        :return:
        """
        gt = self.dataloader.dataset.decode_segmap(gt)
        pred=self.dataloader.dataset.decode_segmap(pred)
        gt = np.array(gt).astype(np.float32).transpose((2, 0, 1))
        gt = torch.from_numpy(gt).type(torch.FloatTensor)
        pred = np.array(pred).astype(np.float32).transpose((2, 0, 1))
        pred = torch.from_numpy(pred).type(torch.FloatTensor)
        img=torch.stack([gt,pred])
        self.summary.visualize_image(writer,title,img,epoch)


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





