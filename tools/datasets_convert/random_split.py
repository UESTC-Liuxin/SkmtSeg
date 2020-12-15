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
    description:随机切分图片
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


class SplitImageSetsByDate(object):
    """
    description:利用日期划分数据集
    """
    def __init__(self,default_split_ratio=0.7,data_root=None):
        self.data_root=data_root
        self.default_split_ratio=default_split_ratio
        section_dict=self.__build_files_table()
        print(section_dict.keys())
        train_set,val_set=self.__split_dataset(section_dict)
        write_path = os.path.join(self.data_root, 'ImageSets')

        self.write_txt(os.path.join(write_path, 'train.txt'), list(train_set))
        self.write_txt(os.path.join(write_path, 'val.txt'), list(val_set))

    def write_txt(self, dist_file, file_list):
        """write file names to distfile
        Args:
            dist_file(str):the abs path of write dist
            file_list:

        Returns:

        """
        file = open(dist_file, 'wb')
        for file_name in file_list:
            file.write((file_name + '\n').encode())
        file.close()

    def __get_section_date(self,file):
        __name = file.split('/')[-1]
        __section=__name.split('_')[0][-2]
        __date =__name.split('_')[1]
        return __section,__date,__name

    def __split_dataset(self,section_dict):
        """

        :param section_dict:
        :return:
        """
        val_set=[]
        train_set=[]
        for section,date_dict in section_dict.items():
            weights=[]
            date_list=date_dict.values()   #获取这个切面下不同日期的图片列表
            for imgs_list in date_list:
                weights.append(len(imgs_list))  #将每个列表的长度进行记录，作为重量
            combines,remain=self.__findBestSets(weights,int(sum(weights)*0.3))

            print('section:',section,'split results:',combines,
                  'over count:',sum(weights),'really count:',int(sum(weights)*0.3-remain))
            for i,imgs_list in enumerate(date_list):
                if(i in combines):
                    val_set+=imgs_list
                else:
                    train_set+=imgs_list
        print(len(train_set),len(val_set))
        return train_set,val_set




    def __findBestSets(self,weights: [list, tuple], C):
        """
        回溯法寻找左逼近于C最好的组合
        :param imgs_counts:
        :param C:
        :return:
        """
        min_sum = C
        bestCombines = []
        weights_count = len(weights)

        def backTrace(start, C, combine):
            nonlocal min_sum, bestCombines  # 一定要声明为非局部变量
            if (C < 0):
                temp = combine[:]  # 复制一份
                i = temp.pop()
                if (min_sum > C + weights[i]):
                    bestCombines = temp
                    min_sum = C + weights[i]
                return
            for i in range(start, weights_count):
                combine.append(i)
                backTrace(i + 1, C - weights[i], combine)
                combine.pop()

        backTrace(0, C, [])
        return bestCombines,min_sum




    def __build_files_table(self):
        """

        :return:a table like {'1':{"200801":['Shoulder11_200720_16.png', 'Shoulder11_200720_17.png']}
                               '2':{"200911":['Shoulder21_200720_16.png', 'Shoulder21_200720_17.png']}
                               }
        """

        section_dict={}
        for file in os.listdir(os.path.join(self.data_root,"SegmentationClass")):
            section,date,name=self.__get_section_date(file)
            if(not section in section_dict.keys()):#如果没有此切面的信息，创建切面下的date和文件列表
                date_dict = {}
                date_dict[date]=[]
                date_dict[date].append(name)
                section_dict[section]=date_dict
            elif (not date in section_dict[section].keys()):#如果这个日期从未出现过
                section_dict[section][date]=[]
                section_dict[section][date].append(name)
            else:
                section_dict[section][date].append(name)
        return section_dict



if __name__ =='__main__':
    data_root = '/home/liuxin/Documents/CV/Project/SKMT/mmsegmentation/data/SKMT/Seg'
    test=SplitImageSetsByDate(data_root=data_root)
