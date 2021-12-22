# -*- coding: utf-8 -*-
"""
@description:

@author: LiuXin
@contact: xinliu1996@163.com
@Created on: 2020/11/27 上午11:19
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None,ignore_index=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.reduction = reduction

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.reshape(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        logpt = F.log_softmax(input,dim=1) # 这里转成log(pt)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = torch.exp(logpt)

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.reduction=="mean": return loss.mean()
        else: return loss.sum()



if __name__ =='__main__':
    from dataloader.skmt import SkmtDataSet
    torch.manual_seed(1000)
    input=torch.rand(2,11,512,512,requires_grad=True)
    target=torch.randint(11,(2,512,512)).long()
    focal=FocalLoss(alpha=torch.softmax(1-torch.Tensor(SkmtDataSet.CLASSES_PIXS_WEIGHTS),dim=0))
    loss=focal(input,target)
    print(loss)









