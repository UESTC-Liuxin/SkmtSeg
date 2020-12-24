# -*- coding: utf-8 -*-
import os
import imageio
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from PIL import Image

class Augment:
    def __init__(self):
        self.seq = iaa.Sequential([
            #iaa.Dropout([0.05, 0.2]),  # drop 5% or 20% of all pixels
            #iaa.Sharpen((0.0, 1.0)),  # sharpen the image
            iaa.Affine(rotate=(-40,40)),  # rotate by -45 to 45 degrees (affects segmaps)
            #iaa.ElasticTransformation(alpha=50, sigma=5)  # apply water effect (affects segmaps)
            #iaa.Crop(px=(0, 16)),  # 对图像进行crop操作，随机在距离边缘的0到16像素中选择crop范围
            iaa.Fliplr(0.5),  # 对百分之五十的图像进行做左右翻转
            #iaa.GaussianBlur((0, 1.0))  # 在模型上使用0均值1方差进行高斯模糊
        ], random_order=True)

    def aug_img(self,image, segmap, seq,):
        ia.seed(1)
        segmap = SegmentationMapsOnImage(np.array(segmap), shape=image.shape)

        images_aug_i, segmaps_aug_i = seq(image=image, segmentation_maps=segmap)
        images_aug_i = Image.fromarray(images_aug_i)

        segmaps_aug_i_ = segmaps_aug_i.get_arr()
        segmaps_aug_i_ = Image.fromarray(np.uint8(segmaps_aug_i_))
        segmaps_aug_i_ = segmaps_aug_i_.convert("P")

        return images_aug_i, segmaps_aug_i_
