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

def array2p_mode(alpha_channel):
    """alpha_channel is a binary image."""
    #assert set(alpha_channel.flatten().tolist()) == {0, 1}, "alpha_channel is a binary image."
    #alpha_channel[alpha_channel == 1] = 128
    h, w = alpha_channel.shape
    image_arr = np.zeros((h, w, 3))
    image_arr[:, :, 0] = alpha_channel
    image_arr[:, :, 1] = alpha_channel
    image_arr[:, :, 2] = alpha_channel
    img = Image.fromarray(np.uint8(image_arr))
    #img_p = img.convert("P")
    return img


def run(image_path, segmap_path, image_aug_path, SegmentationClass_aug_path,txt_set):
    # 1.Load an example image.
    ia.seed(1)
    image = np.array(Image.open(image_path))
    segmap = Image.open(segmap_path)
    segmap = SegmentationMapsOnImage(np.array(segmap), shape=image.shape)
    # 2.Define our augmentation pipeline.
    seq = iaa.Sequential([
        #iaa.Dropout([0.05, 0.2]),  # drop 5% or 20% of all pixels
        #iaa.Sharpen((0.0, 1.0)),  # sharpen the image
        iaa.Affine(rotate=(-40,40)),  # rotate by -45 to 45 degrees (affects segmaps)
        #iaa.ElasticTransformation(alpha=50, sigma=5)  # apply water effect (affects segmaps)
        #iaa.Crop(px=(0, 16)),  # 对图像进行crop操作，随机在距离边缘的0到16像素中选择crop范围
        iaa.Fliplr(0.5),  # 对百分之五十的图像进行做左右翻转
        #iaa.GaussianBlur((0, 1.0))  # 在模型上使用0均值1方差进行高斯模糊


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
        #segmaps_aug_i_ = array2p_mode(segmaps_aug_i_)

        segmaps_aug_i_.save(os.path.join(SegmentationClass_aug_path , name +'.png'))

        # images_aug.append(images_aug_i)
        # segmaps_aug.append(segmaps_aug_i)
        count += 1
    # # 4.show the result
    # cells = []
    # for image_aug, segmap_aug in zip(images_aug, segmaps_aug):
    #     cells.append(image)  # column 1
    #     cells.append(segmap.draw_on_image(image)[0])  # column 2
    #     cells.append(image_aug)  # column 3
    #     cells.append(segmap_aug.draw_on_image(image_aug)[0])  # column 4
    #     cells.append(segmap_aug.draw(size=image_aug.shape[:2])[0])  # column 5
    # # 5.Convert cells to a grid image and save.
    # grid_image = ia.draw_grid(cells, cols=5)
    # imageio.imwrite("example_segmaps.jpg", grid_image)
    return txt_set


if __name__ == "__main__":
    # images_list_path = [os.path.join(images_path, i) for i in os.listdir(images_path)]
    # print(len(images_list_path))
    train_txt_path=os.path.join(txt_path, 'train_1.txt')
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
    f = open(txt_path, 'w')
    f.write(txt_set)
    f.close()