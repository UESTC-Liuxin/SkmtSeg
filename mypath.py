# -*- coding: utf-8 -*-
"""
@description:

@author: LiuXin
@contact: xinliu1996@163.com
@Created on: 2020/11/9 上午10:42
"""

class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/media/Program/CV/dataset/VOCdevkit/'  # folder that contains VOCdevkit/.
        elif dataset == 'skmt':
            return 'data/Synapse/'
            #return 'data/SKMT/Seg/'
        elif dataset == 'synapse':
            return 'data/Synapse/'
        elif dataset == 'CAMUS':
            return 'data/CAMUS'
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
