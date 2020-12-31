# -*- coding: utf-8 -*-
import os
import imageio
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from PIL import Image

class Augment(object):
    def __init__(self):
        self.seq = iaa.Sequential([
             iaa.Noop(),
             #iaa.Sharpen((0.0, 1.0)),  # sharpen the image
             #iaa.GammaContrast((0.5, 2.0)),  # 对比度增强
             #iaa.Alpha((0.0, 1.0), iaa.HistogramEqualization()),  # 直方图均衡
             #iaa.Affine(rotate=(-20, 20)),  # rotate by -40 to 40 degrees (affects segmaps)
             #iaa.Fliplr(0.5)  # 对百分之五十的图像进行做左右翻
        ], random_order=True)

    def __call__(self, sample):
        image =  np.array(sample['image'])
        segmap = sample['label']
        ia.seed(1)
        segmap = SegmentationMapsOnImage(np.array(segmap), shape=image.shape)

        images_aug_i, segmaps_aug_i = self.seq(image=image, segmentation_maps=segmap)
        images_aug_i = Image.fromarray(images_aug_i)

        segmaps_aug_i_ = segmaps_aug_i.get_arr()
        segmaps_aug_i_ = Image.fromarray(np.uint8(segmaps_aug_i_))
        segmaps_aug_i_ = segmaps_aug_i_.convert("P")

        return {'image': images_aug_i,
                'label': segmaps_aug_i_}
