# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2
from dataloader.skmt import SkmtDataSet
from dataloader.transforms_utils import augment as au
import copy
dataset_root = 'data/SKMT/Seg'
images_path = os.path.join(dataset_root, 'JPEGImages')
lable_path =  os.path.join(dataset_root, 'SegmentationClass')
save_path = 'sav_pro'
save_path_seg='sav_seg'
save_path_img='sav_img'
def decode_segmap(label_mask):
    label_colours = SkmtDataSet.PALETTE
    n_classes = len(label_colours)

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return rgb
def encode_segmap( mask):
    """Encode segmentation label images as pascal classes

    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.

    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)

    for ii, label in enumerate(SkmtDataSet.PALETTE):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask
def preprocessing(img,lable):
    """
    preprocessing 裁剪图片
    """
    # img=cv2.GaussianBlur(img,(5,5),0)
    # mask=np.zeros(img.shape,dtype=np.uint8)
    augm = au.Augment()
    # sample = augm(sample)
    ret, binary = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)  # 二值图
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 边缘检测函数
    contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)
    # mask=cv2.drawContours(mask, contours, 0, (255, 0, 255), 3)  # 绘画出边缘
    top=(min(row[0][0] for row in contours[0]),min(row[0][1] for row in contours[0]))
    bottom = (max(row[0][0] for row in contours[0]),max(row[0][1] for row in contours[0]))
    img=img[top[1]:bottom[1],top[0]:bottom[0]]
    lable = lable[top[1]:bottom[1], top[0]:bottom[0]]
    return img,lable


def resize_(img, min_side=512):
    size = img.shape
    h, w = size[0], size[1]
    # 长边缩放为min_side
    scale = max(w, h) / float(min_side)
    new_w, new_h = int(w / scale), int(h / scale)
    resize_img = cv2.resize(img, (new_w, new_h))
    # 填充至min_side * min_side
    if new_w % 2 != 0 and new_h % 2 == 0:
        top, bottom, left, right = (min_side - new_h) / 2, (min_side - new_h) / 2, (min_side - new_w) / 2 + 1, (
                    min_side - new_w) / 2
    elif new_h % 2 != 0 and new_w % 2 == 0:
        top, bottom, left, right = (min_side - new_h) / 2 + 1, (min_side - new_h) / 2, (min_side - new_w) / 2, (
                    min_side - new_w) / 2
    elif new_h % 2 == 0 and new_w % 2 == 0:
        top, bottom, left, right = (min_side - new_h) / 2, (min_side - new_h) / 2, (min_side - new_w) / 2, (
                    min_side - new_w) / 2
    else:
        top, bottom, left, right = (min_side - new_h) / 2 + 1, (min_side - new_h) / 2, (min_side - new_w) / 2 + 1, (
                    min_side - new_w) / 2

    pad_img = cv2.copyMakeBorder(resize_img, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT,
                                 value=[0,0,0])  # 从图像边界向上,下,左,右扩的像素数目
    return pad_img

if __name__ == '__main__':
    images_list_path = [os.path.join(images_path, i) for i in os.listdir(images_path)]
    print(len(images_list_path))
    for count, image_path in enumerate(images_list_path):
        print('{}image'.format(count))
        print(image_path)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        image_name = image_path.split("/")[-1].split(".")[-2]
        lable_paths= os.path.join(lable_path,image_name+'.png')
        print(lable_paths)
        lable = cv2.imread(lable_paths)
        lable = cv2.cvtColor(lable, cv2.COLOR_RGB2GRAY)
        # lable = decode_segmap(lable)

        img_1,lable_1=preprocessing(img,lable)
        # img =resize_(img)
        # lable = resize_(lable)
        # img_1 = resize_(img_1)
        # lable_1 = resize_(lable_1)

        lable = Image.fromarray(np.uint8(lable))
        lable_1 = Image.fromarray(np.uint8(lable_1))
        img = Image.fromarray(np.uint8(img))
        img_1 = Image.fromarray(np.uint8(img_1))

        img_1.save(os.path.join(save_path_img, image_name + '.jpg'))
        lable_1.save(os.path.join(save_path_seg, image_name + '.png'))
        # lable = cv2.cvtColor(lable, cv2.COLOR_RGB2GRAY)
        #
        # lable_1 = cv2.cvtColor(lable_1, cv2.COLOR_RGB2GRAY)
        # lable_1 = decode_segmap(lable_1)
        # rrr = Image.new('RGB', (2048, 512), (255, 255, 255))
        # rrr.paste(img, (0, 0))  # 从0，0开始贴图
        # rrr.paste(img_1, (512, 0))  # 从0，0开始贴图
        # rrr.paste(lable, (1024, 0))
        # rrr.paste(lable_1, (1536, 0))
        # rrr.save(os.path.join(save_path, image_name+'.png'))

