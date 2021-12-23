# -*- coding: utf-8 -*-
"""
@description:

@author: LiuXin
@contact: xinliu1996@163.com
@Created on: 2020/11/12 下午7:07
"""
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2
from dataloader.skmt import SkmtDataSet
dataset_root = '../data/SKMT/Seg'
images_path = os.path.join(dataset_root, 'SegmentationClass')



def find_pic(img, array_list, pixs,n_class):
    img_sum = np.sum(img == array_list, axis=-1)
    pix_numbers = img_sum.reshape(-1).tolist().count(3)
    if pix_numbers:
        pixs += pix_numbers
        n_class += 1
    return pixs, n_class

def compute_class(all_class, n_class):
    return n_class / all_class

def frequence():
    images_list_path = [os.path.join(images_path, i) for i in os.listdir(images_path)]
    print(len(images_list_path))

    class_pixs = np.zeros(len(SkmtDataSet.CLASSES))
    clss_n = np.zeros(len(SkmtDataSet.CLASSES))
    f_class = np.zeros(len(SkmtDataSet.CLASSES))

    for count, image_path in enumerate(images_list_path):
        print('{}image'.format(count))
        img = cv2.imread(image_path)
        '''for h in range(0, img.shape[0]):
            for w in range(0, img.shape[1]):
                (b,g,r) = img[h,w]
                if (b,g,r) != (0, 0, 0):
                    print((b,g,r))'''
        for ii, label in enumerate(SkmtDataSet.CLASSES):
            #if ii != 0:
            class_pixs[ii], clss_n[ii] = find_pic(img, [ii,ii,ii], class_pixs[ii], clss_n[ii])
    all_class = np.sum(clss_n)
    for ii, label in enumerate(SkmtDataSet.CLASSES):
        f_class[ii] = compute_class(all_class,clss_n[ii])

    for ii, label in enumerate(SkmtDataSet.CLASSES):
        print('{}: pixs:{:.0f} num:{:.0f}  frequent:{:.2f}'.format(label,class_pixs[ii], clss_n[ii],f_class[ii]))

    f_class_median = np.median(np.array(f_class))
    print(f_class_median)
    print(f_class_median / np.array(f_class))

    f_class_median = np.median(np.array(class_pixs))
    print(f_class_median)
    print(f_class_median / np.array(class_pixs))


if __name__ == '__main__':
    frequence()
