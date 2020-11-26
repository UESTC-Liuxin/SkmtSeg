# -*- coding: utf-8 -*-
"""
@author: LiuXin
@contact: xinliu1996@163.com
@Created on: 2020/8/4 下午1:56
"""

from pycocotools.coco import COCO
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os,shutil,cv2
from dataloader.skmt import SkmtDataSet


def resave_img(img_name,output_path,prefix):
    '''

    Args:
        img_name: the img_name containing path
        output_path:
        prefix: some info to distinguish img

    Returns:

    '''
    name_splits=img_name.split('/')
    img=Image.open(img_name)
    if img.mode != 'RGB':
        img=img.convert('RGB')
    img.save(os.path.join(output_path,'JPEGImages',prefix+name_splits[-1]))
    # shutil.copyfile(img_name,os.path.join(output_path,'JPEGImages',prefix+name_splits[-1]))



def get_mask(coco,anns,file_name,prefix=None):
    '''read img file and construct a (w,h) seg_mask according to the anns

    Args:
        anns(list[dict]): the anns of one img
        input_root:
        img_info(dict):
            eg: {'id': 1, 'license': 0, 'file_name': '1.jpg',...,'width': 755}

    Returns:
        seg_map(np.array(w,h)):the map of class_id
    '''
    img = Image.open(file_name)
    seg_mask = np.zeros((img.size[1], img.size[0]), dtype=np.int)
    for ann in anns:
        mask = coco.annToMask(ann)
        seg_mask[np.where(mask)]=ann['category_id']
        # return mask
    seg_map = Image.fromarray(seg_mask.astype('uint8')).convert('P')

    return seg_map



def ann_to_segmap(coco,input_root,output_root,prefix):
    '''
    get anns of each img and build a seg_map ,and save it to the seg_map_path

    Args:
        coco(COCO): obj of coco
        input_root:
        output_root:
        prefix: the prefix of newname for images,ig:Shoulder11_200812_1.jpg
    Returns:
        None
    '''
    imgIds = coco.getImgIds()
    img_infos = coco.loadImgs(imgIds)
    img_path =os.path.join(input_root,'images')
    seg_map_path=os.path.join(output_root,'SegmentationClass')
    #加载所有图片
    for index,img_info in enumerate(img_infos):
        annIds = coco.getAnnIds(imgIds=img_info['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)
        file_name=os.path.join(img_path,img_info['file_name'])
        seg_map=get_mask(coco,anns,file_name)
        #添加文件名前缀并保存分割图
        if prefix:
            resave_img(file_name,output_root,prefix)

        seg_map.save(os.path.join(seg_map_path,prefix+img_info['file_name'].replace('jpg','png')))

    return None


def visualize_map_img(img_path,seg_map_path):
    '''visualize the seg map by rgb picture

    Returns:
        None

    '''
    img_files=os.listdir(seg_map_path)
    for file in img_files:
        mask=Image.open(os.path.join(seg_map_path,file))
        img=Image.open(os.path.join(img_path,file.replace('png','jpg')))
        label_mask=np.array(mask)
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, np.max(label_mask)+1):
            r[label_mask == ll] = SkmtDataSet.PALETTE[ll][ 0]
            g[label_mask == ll] = SkmtDataSet.PALETTE[ll][1]
            b[label_mask == ll] = SkmtDataSet.PALETTE[ll][2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        #plot img
        plt.figure(figsize=(200,100))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.subplot(1, 2, 2)
        plt.imshow(rgb)
        plt.show()



if __name__ =='__main__':
    img_path='/home/liuxin/Documents/CV/dataset/SKMT/Seg/JPEGImages'
    seg_map_path='/home/liuxin/Documents/CV/dataset/SKMT/Seg/SegmentationClass'
    visualize_map_img(img_path,seg_map_path)