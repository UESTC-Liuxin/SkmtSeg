# -*- coding: utf-8 -*-
"""
@description:

@author: Maglan
@contact: maglanyulan@163.com
@Created on: 2021/4/11 下午4:22
"""
from dataloader.callosum import CallDataSet
from dataloader.skmt import SkmtDataSet
from dataloader.synapse import Synapse_dataset


def build_dataset(args):
    """build a loss  by mode

    :param mode:loss type,str:"ce","focal","iou",etc
    :param args:other para pass to Loss Module
    :param kwargs:
    :return:
    """
    if(args.dataset=='synapse'):
        train_set =Synapse_dataset(args,split='train')
        val_set = Synapse_dataset(args, split='val')
        return train_set, val_set
    elif (args.dataset == 'call'):
        train_set = CallDataSet(args, split='train')
        val_set = CallDataSet(args, split='val')
        return train_set, val_set
    elif (args.dataset == 'skmt'):
        train_set = SkmtDataSet(args,split='train')
        val_set = SkmtDataSet(args, split='val')
        return train_set,val_set
    else:
        raise NotImplementedError

