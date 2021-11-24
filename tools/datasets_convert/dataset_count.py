# -*- coding: utf-8 -*-
"""
@description:

@author: LiuXin
@contact: xinliu1996@163.com
@Created on: 2020/11/12 下午7:07
"""

from pycocotools.coco import COCO, maskUtils
from dataloader.skmt import  SkmtDataSet
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2

dataset_root = '/home/liuxin/Documents/CV/Project/SkmtSeg/data/SKMT/Seg'
dataType = 'default'
annFile = '{}/annotations/instances_{}.json'.format(dataset_root, dataType)
img_path = os.path.join(dataset_root, 'JPEGImages')
seg_img_path = os.path.join(dataset_root, 'SegmentationClass')


# 读取coco文件
# coco = COCO(annFile)


def count_rgb(imgs_path):
    '''count all the mean and std of RGB channels of origin imge
    Args:
        imgs_path:the path of origin imgs

    Returns:
        mean_dict:the mean value of RGB
        std_dict: as shown on
    '''
    file_names = os.listdir(imgs_path)
    category = ['R', 'G', 'B']
    mean_dict = dict(zip(category, [0] * 3))
    std_dict = dict(zip(category, [0] * 3))
    per_image_Rmean = []
    per_image_Gmean = []
    per_image_Bmean = []
    # notice：CV2：BGR
    for file_name in file_names:
        img = cv2.imread(os.path.join(imgs_path, file_name), 1)
        per_image_Bmean.append(np.mean(img[:, :, 0]))
        per_image_Gmean.append(np.mean(img[:, :, 1]))
        per_image_Rmean.append(np.mean(img[:, :, 2]))
    mean_dict['R'] = np.mean(per_image_Rmean)
    mean_dict['G'] = np.mean(per_image_Gmean)
    mean_dict['B'] = np.mean(per_image_Bmean)
    std_dict['R'] = np.std(per_image_Rmean)
    std_dict['G'] = np.std(per_image_Gmean)
    std_dict['B'] = np.std(per_image_Bmean)

    return mean_dict, std_dict


def count_anns_category(coco):
    '''count the category distribution of annotations

    Args:
        coco:the obj of COCO

    Returns:
        countor_dict: the dict of
    '''
    anns = coco.loadAnns(coco.getAnnIds())
    countor_values = [0] * len(SkmtDataSet.CLASSES)
    for ann in anns:
        countor_values[ann['category_id']] += 1
    countor_dict = dict(zip(SkmtDataSet.CLASSES, countor_values))
    return countor_dict






if __name__ == '__main__':
    #countor_dict=count_anns_category(coco)
    #print(countor_dict)
    #x=[index for index in range(len(countor_dict))]
    #y=countor_dict.values()
    #plt.bar(x,y,label='the dist of cate')
    #plt.show()
    mean_dict, std_dict = count_rgb(img_path)
    print('mean:', mean_dict)
    print('std:', std_dict)
