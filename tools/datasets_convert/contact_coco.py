# -*- coding: utf-8 -*-
"""
@description:
Combine all the coco sub-data sets and follow the format of
{JointSectionNumber_date_index}(Shoulder5_200805_0001.img)

@author: LiuXin
@contact: xinliu1996@163.com
@Created on: 2020/8/6 上午9:40
"""
import os
from pycocotools.coco import COCO
from tools.datasets_convert.coco2voc import ann_to_segmap
#
input_root='/home/liuxin/Documents/CV/dataset/SKMT/COCO'
output_root='/home/liuxin/Documents/CV/dataset/SKMT/Seg'






def add_subdataset(subdataset_list,cwd):
    '''

    Args:
        subdataset_list:
        cwd: a unicode string representing the current working directory,a abs path

    Returns:

    '''
    dir_list=os.listdir(cwd)
    if not dir_list :
        return
    for key in dir_list:
        dir = os.path.join(cwd, key)
        if os.path.isfile(dir):
            continue
        if ('skmt' in dir):
            subdataset_list.append(dir)
            continue
        add_subdataset(subdataset_list,dir)

def search_subdataset(data_root):
    """
    search all sub root and get info of them;
    build the img prefix
    Args:
        data_root:

    Returns:

    """

    #First level directory
    subdataset_list=[]
    add_subdataset(subdataset_list,data_root)
    return subdataset_list

def contact_dataset(input_root,output_root):
    '''
    search all sub coco dataset in input_root,and contact it to a list;
    according to the dataset info to rename images(ig:Shoulder11_200812_1.jpg) and save it.
    Args:
        data_root:

    Returns:

    '''
    dataset_list=search_subdataset(input_root)
    for i in dataset_list:
        annFile=os.path.join(i,'annotations','instances_default.json')
        coco = COCO(annFile)
        name_split=i.split('/')
        joint=name_split[-3]
        section=name_split[-1].replace('skmt','')
        date=name_split[-2]
        prefix=joint+section+'_'+date+'_'
        ann_to_segmap(coco=coco,
                      input_root=i,
                      output_root=output_root,
                      prefix=prefix)






if __name__ == '__main__':

    # a = prefix_dict = Dict()
    # search_subdataset(input_root)
    contact_dataset(input_root, output_root)

