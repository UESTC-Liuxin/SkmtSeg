import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class FocalLoss(nn.Module):
    def __init__(self,weight=None, size_average=True, ignore_index=255,gamma=2, alpha=0.5):
        super(FocalLoss,self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.gamma = gamma
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)


    def forward(self,logit, target):
        n, c, h, w = logit.size()

        logpt = -self.ce_loss(logit, target.long())
        pt = torch.exp(logpt)
        if self.alpha is not None:
            logpt *= self.alpha
        loss = -((1 - pt) ** self.gamma) * logpt

        if self.size_average:
            loss /= n

        return loss


class Loss(nn.Module):

    def __init__(self, args,mode='ce',weight=None, size_average=True, ignore_index=255):
        super(Loss,self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.args = args

        #两个参数
        self.sigma = nn.Parameter(torch.rand([1]))
        self.beta = nn.Parameter(torch.rand([1]))


        if(self.args.auxiliary is not None):
            self.loss_auxiliary = self.build_loss(mode=mode)
        self.loss_trunk = self.build_loss(mode=mode)

    def build_loss(self,mode):
        if mode == 'ce':
            return nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        elif(mode =='focal'):
            return FocalLoss()

        else:
            raise NotImplementedError

    def forward(self, preds,label):

        loss2 = self.loss_trunk(preds['trunk_out'], label['label'])
        if(self.args.auxiliary is not None):
            loss1=self.loss_auxiliary(preds['auxiliary_out'],label['label'])
            loss = (loss1 + loss2).mean()
        else:
            loss =loss2.mean()

        return loss



if __name__ =="__main__":
    inputs=torch.rand(2,19,512,512)
    target=torch.randint(high=12,size=(2,512,512))
    criterion=FocalLoss()
    loss=criterion(inputs,target)
    print(loss)





