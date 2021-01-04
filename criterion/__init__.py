# -*- coding: utf-8 -*-
"""
@description:

@author: LiuXin
@contact: xinliu1996@163.com
@Created on: 2020/11/27 上午11:18
"""
import torch
import torch.nn as nn

from criterion.WCELoss import WCELoss
from criterion.FocalLoss import FocalLoss
from criterion.DiceLoss import DiceLoss
from criterion.IouLoss import IouLoss
from criterion.LabelSmoothCELoss import LabelSmoothCrossEntropy2D
from criterion.ComposeLoss import ComposeLoss,CaculateTLoss



def build_criterion(auxiliary=None,trunk=None):
    """
    define losses by a dict
    @example:
    CRITERION=dict(
        auxiliary=dict(
            losses=dict(ce=dict(reduction='mean')),
            loss_weights=1
        ),
        trunk=dict(
            losses=dict(
                ce=dict(reduction='mean'),
                dice=dict(smooth=1, p=2,reduction='mean')
            ),
            loss_weights=[0.5,0.5]
        )
    )
    criterion=build_criterion(CRITERION)
    :param auxiliary:
    :param trunk:
    :return:
    """
    trunk_losses=[]
    for mode, kwargs in trunk['losses'].items():
        trunk_losses.append(build_loss(mode=mode, **kwargs))

    trunk_loss_weights = trunk['loss_weights']
    assert len(trunk_losses)==len(trunk_loss_weights)
    trunk_compose=ComposeLoss(trunk_losses,trunk_loss_weights)

    if(auxiliary is not None):
        auxiliary_losses=[]
        for mode,kwargs in auxiliary['losses'].items():
            auxiliary_losses.append(build_loss(mode=mode,**kwargs))
        auxiliary_loss_weights=auxiliary['loss_weights']
        assert len(auxiliary_losses)==len(auxiliary_loss_weights)
        auxiliary_compose=ComposeLoss(auxiliary_losses,auxiliary_loss_weights)
        return CaculateTLoss(trunk_compose,auxiliary_compose)

    else:
        return CaculateTLoss(trunk_compose)




def build_loss(mode,*args,**kwargs):
    """build a loss  by mode

    :param mode:loss type,str:"ce","focal","iou",etc
    :param args:other para pass to Loss Module
    :param kwargs:
    :return:
    """
    if(mode=='ce'):
        return WCELoss(*args,**kwargs)
    elif(mode =='focal'):
        return FocalLoss(*args,**kwargs)
    elif(mode =='iou'):
        return IouLoss(*args,**kwargs)
    elif(mode=='dice'):
        return DiceLoss(*args,**kwargs)
    elif(mode=='smoothce'):
        return LabelSmoothCrossEntropy2D(*args,**kwargs)
    else:
        raise NotImplementedError

if __name__ == "__main__":

    img=torch.rand(2,6,512,512)
    gt =torch.randint(6,(2,512,512)).long()
    CRITERION = dict(
        auxiliary=dict(
            losses=dict(
                ce=dict(reduction='mean'),
                dice=dict(smooth=1, p=2, reduction='mean')
            ),
            loss_weights=[0.5, 0.5]
        ),
        trunk=dict(
            losses=dict(
                # ce=dict(reduction='mean'),
                dice=dict(smooth=1, p=2, reduction='mean')
            ),
            loss_weights=[0.5, 0.5]
        )
    )
    criterion=build_criterion(**CRITERION)
    print(criterion({'auxiliary':img,'trunk':img},gt))

