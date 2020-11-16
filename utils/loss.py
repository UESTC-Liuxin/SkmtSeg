import torch
import torch.nn as nn



class FocalLoss(nn.Module):
    def __init__(self,gamma=2, alpha=0.5):
        super(FocalLoss,self).__init__()

        self.gamma=gamma
        self.alpha=alpha
        self.ce_loss = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)


    def forward(self,logit, target):
        n, c, h, w = logit.size()

        logpt = -self.ce_loss(logit, target.long())
        pt = torch.exp(logpt)
        if self.alpha is not None:
            logpt *= self.alpha
        loss = -((1 - pt) ** self.gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

class Loss(nn.Module):

    def __init__(self, args,mode='ce',weight=None, size_average=True, batch_average=True, ignore_index=255):
        super(Loss,self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average

        #两个参数
        self.sigma = nn.Parameter(torch.rand([1]))
        self.beta =nn.Parameter(torch.rand([1]))
        self.args=args

        if(self.args.auxiliary is not None):
            self.loss_auxiliary=self.build_loss(mode=mode)
        self.loss_trunk=self.build_loss(mode=mode)

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







