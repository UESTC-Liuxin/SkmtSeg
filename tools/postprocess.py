# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2
from dataloader.skmt import SkmtDataSet

dataset_root = '../data/CAMUS'
images_path = os.path.join(dataset_root, 'SegmentationClass')

def find_pic(img,ii):
    img1 = np.zeros(img.shape, np.uint8)
    for h in range(0, img.shape[0]):
        for w in range(0, img.shape[1]):
            r = img[h, w]
            if r == ii :
                img1[h, w] = 255
            else:
                img1[h, w] = 0
    return img1

def cnt_area(cnt):
    area = cv2.contourArea(cnt)
    return area
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
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3), dtype=np.uint8)
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

def postprocess(img, classnum):
    # print(img.shape)
    img = cv2.copyMakeBorder(img, 50,50,50,50, cv2.BORDER_CONSTANT,value=0)
    # print(img.shape)
    h, w = img.shape
    temp = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    for ii in range(1,classnum):
        img1 = find_pic(img,ii)
        ret, thresh = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours.sort(key=cnt_area, reverse=True)
        # 画出轮廓：temp是黑色幕布，contours是轮廓，-1表示全画，然后是颜色，厚度
        for c in contours:
            area = cv2.contourArea(c)
            if area > img.shape[0]*img.shape[1]/2 :continue
            # 分别在复制的图像上和白色图像上绘制当前轮廓
            cv2.drawContours(temp, [c], 0, (ii,ii,ii), thickness=-1)
            break
    temp = cv2.cvtColor(temp[50:h-50, 50:w-50], cv2.COLOR_BGR2GRAY)
    # print(temp.shape)
    return temp

def get_section(name):
    print(name)
    _name = name.split('/')[-1]
    _section = _name.split('_')[0][-2]
    print(_section)
    return int(_section)



if __name__ == '__main__':
    images_list_path = [os.path.join(images_path, i) for i in os.listdir(images_path)]
    print(len(images_list_path))
    for count, image_path in enumerate(images_list_path):
        print('{}image'.format(count))
        img = cv2.imread(image_path)
        # get_section(image_path)
        #img = encode_segmap(img)
        #print(img.mode)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img1 = decode_segmap(img)

        img1 = Image.fromarray(np.uint8(img1))
        img1.save(os.path.join(dataset_root, str(count) + '.png'))

        temp = postprocess(img, 11)
        temp = decode_segmap(temp)
        temp = Image.fromarray(np.uint8(temp))
        #temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)

        temp.save(os.path.join(dataset_root, str(count) + '_new.png'))



