# -*- coding: utf-8 -*-
import os
import imageio
import numpy as np
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from PIL import Image

dataset_root = '/media/E/liuxin/SKMT/Seg'
images_path = os.path.join(dataset_root, 'JPEGImages')
segmaps_path = os.path.join(dataset_root, 'SegmentationClass')
txt_path=os.path.join(dataset_root, 'ImageSets')

#图片增强后保存文件夹
image_aug_path = os.path.join(dataset_root,'JPEGImages_AUG')
SegmentationClass_aug_path = os.path.join(dataset_root,'SegmentationClass_AUG')

def run(image_path, segmap_path, image_aug_path, SegmentationClass_aug_path,txt_set):
    # 1.Load an example image.
    ia.seed(1)
    image = np.array(Image.open(image_path))
    segmap = Image.open(segmap_path)
    segmap = SegmentationMapsOnImage(np.array(segmap), shape=image.shape)
    # 2.Define our augmentation pipeline.
    seq = iaa.Sequential([
        iaa.Sharpen((0.0, 1.0)),  # sharpen the image
        iaa.GammaContrast((0.5, 2.0)),  # 对比度增强
        iaa.Alpha((0.0, 1.0), iaa.HistogramEqualization()),  # 直方图均衡
        iaa.Affine(rotate=(-40, 40)),  # rotate by -40 to 40 degrees (affects segmaps)
        iaa.Fliplr(0.5)  # 对百分之五十的图像进行做左右翻
    ], random_order=True)
    file_name = image_path.split("/")[-1]
    file_name = file_name.split(".")[-2]

    count = 1

    for _ in range(5):
        name = file_name +'_'+ f"{count:04d}"
        #print(name)
        txt_set= txt_set+name+'\n'
        images_aug_i, segmaps_aug_i = seq(image=image, segmentation_maps=segmap)
        images_aug_i = Image.fromarray(images_aug_i)
        images_aug_i.save(os.path.join(image_aug_path , name +'.jpg'))

        segmaps_aug_i_ = segmaps_aug_i.get_arr()
        segmaps_aug_i_ = Image.fromarray(np.uint8(segmaps_aug_i_))
        segmaps_aug_i_ = segmaps_aug_i_.convert("P")

        segmaps_aug_i_.save(os.path.join(SegmentationClass_aug_path , name +'.png'))
        count += 1

    return txt_set


if __name__ == "__main__":
    # images_list_path = [os.path.join(images_path, i) for i in os.listdir(images_path)]
    # print(len(images_list_path))
    train_txt_path=os.path.join(txt_path, 'train.txt')
    txt_set = ""
    with open(train_txt_path, 'r') as fr:
        for line in fr.readlines():
            txt_set=txt_set+line
            image_path=os.path.join(images_path,line.strip()+'.jpg')
            #print('{}image'.format(line))
            segmap_path = os.path.join(segmaps_path,line.strip()+'.png')
            print(segmap_path,image_path)
            txt_set=run(image_path, segmap_path, image_aug_path, SegmentationClass_aug_path,txt_set)
    txt_path = os.path.join(txt_path, 'train_aug.txt')
    # f = open(txt_path, 'w')
    # f.write(txt_set)
    # f.close()