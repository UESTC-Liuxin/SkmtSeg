# -*- coding: utf-8 -*-
"""
@description:modified from RanTran.py by cly.

@author: LiuXin
@contact: xinliu1996@163.com
@Created on: 2020/8/13 上午9:35
"""
import os
import numpy as np

class SplitImageSets(object):
    """
    description:
    """
    def __init__(self,default_split_ratio=0.8,joint_key='Shoulder',data_root=None):
        self.__files_table={}
        self.__split_ratio_table={}
        self.data_root=data_root
        self.default_split_ratio=default_split_ratio
        self.joint_key=joint_key
        self.__build_files_table()


    def __build_files_table(self):
        """build files table according to the section_num

        Returns:

        """
        file_list =os.listdir(os.path.join(self.data_root,'JPEGImages'))
        for file_name in file_list:
            section_num=file_name.replace(self.joint_key,'')[0]
            if not section_num.isdigit():
                raise NameError(f'section num must be a digit,but got {section_num}')
            if not section_num in self.__files_table:
                self.__files_table[section_num]=[]
            self.__files_table[section_num].append(
                file_name.replace('.jpg','').replace('.png',''))
            self.__split_ratio_table[section_num]=self.default_split_ratio

    # TODO: here should be more rigorous,because set is not an ordered data structure
    def random_split(self,shuffle=True,random_seed=None):
        """

        Args:
            shuffle:invalid para,because use set
            random_seed: invalid para,because use set

        Returns:

        """
        val_set=set()
        train_set=set()
        if random_seed:
            np.random.seed(random_seed)

        for (key,file_list) in self.__files_table.items():
            sample_num=round(len(file_list) * self.__split_ratio_table[key])
            #随机不重复采样
            trian_temp_set = set(np.random.choice(file_list,sample_num,replace=False))
            val_temp_set = set(file_list)-trian_temp_set
            #合并集合
            train_set=train_set | trian_temp_set
            val_set =val_set | val_temp_set
        assert (len(train_set)+len(val_set))==len(
            os.listdir(os.path.join(self.data_root,'JPEGImages')))
        #write to file
        write_path=os.path.join(self.data_root,'ImageSets')
        self.write_txt(os.path.join(write_path,'train.txt'),list(train_set))
        self.write_txt(os.path.join(write_path, 'val.txt'), list(val_set))

    def write_txt(self,dist_file,file_list):
        """write file names to distfile
        Args:
            dist_file(str):the abs path of write dist
            file_list:

        Returns:

        """
        file=open(dist_file,'wb')
        for file_name in file_list:
            file.write((file_name+'\n').encode())
        file.close()

    @property
    def files_table(self):
        return self.__files_table

    @property
    def split_ratio_table(self):
        return self.__split_ratio_table

    @property
    def split_ratio(self,key):
        return self.__split_ratio_table[key]

    @split_ratio.setter
    def split_ratio(self,arg):
        """set split_ratio in __split_ratio_table
        Args:
            args:((key(str),value(float)),...)
        Returns:None
        @example:
            imgset=SplitImageSets(data_root=data_root)
            imgset.split_ratio=('1',0.9),('2',0.7)

        """
        for key,value in arg:
            if not isinstance(value,float):
                raise TypeError(f'value must be float,but got {type(value)}')
            if not 0<value<1:
                raise NameError(f'value must between 0 and 1,but got {value}')
            if not (key in self.__split_ratio_table):
                raise Exception("Invalid key!", key)
            self.__split_ratio_table[key]=value


if __name__ =='__main__':
    data_root = '/home/liuxin/Documents/CV/Project/SKMT/mmsegmentation/data/SKMT/Seg'
    imgset=SplitImageSets(data_root=data_root)
    # imgset.split_ratio=('1',0.9),('2',0.7)
    # print(imgset.split_ratio_table)
    imgset.random_split(random_seed=1)
