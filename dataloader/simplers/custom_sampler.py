# -*- coding: utf-8 -*-
"""
@description:

@author: LiuXin
@contact: xinliu1996@163.com
@Created on: 2020/12/15 下午2:49
"""
import math
import torch
from torch.utils.data.sampler import Sampler
from torch._six import int_classes as _int_classes


class CustomRandomSampler(Sampler):
    """

    """

    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size=batch_size
        self.batch_idx=[]
        idx_list=list(range(len(self.data_source)))
        start=0
        self.table=self.data_source.count_section()
        #TODO:此处的代码冗余，逻辑不够清晰，有时间重新优化
        for k,v in self.table.items():
            #如果此切面的数量小于1，舍去此切面
            if(v<1):
                continue
            for idx in range(0,v,self.batch_size):
                # 如果超过了最高索引值，就从后往前取值
                idx+=start
                max_index=start+v-1
                if(idx+batch_size>max_index+2):
                    self.batch_idx.append(idx_list[max_index+1-self.batch_size:max_index+1])
                else:
                    self.batch_idx.append(idx_list[idx:idx+self.batch_size])
            start+=v


    def __iter__(self):
        n=len(self.batch_idx)
        return iter(torch.randperm(n).tolist())

    def __len__(self):
        return len(self.batch_idx)

class BatchSampler(Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))

        self.sampler = sampler

    def __iter__(self):
        for idx in self.sampler:
            batch=self.sampler.batch_idx[idx]
            yield batch

    def __len__(self):
        return len(self.sampler)

