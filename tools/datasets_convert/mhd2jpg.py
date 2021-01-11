import cv2
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
dataset_root = '../../data/CAMUS/'

def plot_ct_scan(scan, num_column=4, jump=1):
    num_slices = len(scan)
    num_row = (num_slices//jump + num_column - 1) // num_column
    f, plots = plt.subplots(num_row, num_column, figsize=(num_column*5, num_row*5))
    for i in range(0, num_row*num_column):
        plot = plots[i % num_column] if num_row == 1 else plots[i // num_column, i % num_column]
        plot.axis('off')
        if i < num_slices//jump:
            plot.imshow(scan[i*jump], cmap=plt.cm.bone)

#创建空目录
def make_dir(save_path):
	if not os.path.isdir(save_path):
		os.makedirs(save_path)

def save_img(images_path,save_path):
    save_path_gt = os.path.join(save_path, 'SegmentationClass')
    save_path = os.path.join(save_path, 'JPEGImages')

    make_dir(save_path)
    make_dir(save_path_gt)
    mhdpaths = os.listdir(images_path)

    txt = ''
    for mhdpath in mhdpaths:
        if mhdpath.find('mhd')>=0:
            data =sitk.ReadImage(os.path.join(images_path,mhdpath))
            spacing = data.GetSpacing()
            scan = sitk.GetArrayFromImage(data)
            # print('scan.shape', scan.shape)  # 图像大小
            # print('spacing: ', spacing)  # 间隔
            # print('# slice: ', len(scan))  # 切片数量

            for i in range(scan.shape[0]):
                im = Image.fromarray(np.uint8(scan[i, :, :]))  # 这里就是提取的图片数据
                imname=mhdpath.split('.')[0]
                sp=imname.split('_')
                if sp[-1]=='gt':
                    name=sp[0] +'_'+sp[1]+'_'+sp[2]
                    im.save(save_path_gt + '/' + name+ '.png')  # 保存gt图片
                    txt += name + '\n'
                else:
                    im.save(save_path + '/' + imname + '.jpg')  # 保存图片
    return txt

if __name__ == '__main__':

    txt_path = os.path.join(dataset_root, 'ImageSets')
    save_path = os.path.join(dataset_root,'JPEGImages')
    make_dir(txt_path)

    train_path = os.path.join(dataset_root, 'training')
    paths = os.listdir(train_path)
    txt=''
    for path in paths:
        txt+=save_img(os.path.join(train_path,path), dataset_root)
    f = open(os.path.join(txt_path,'train.txt'), 'w')
    f.write(txt)
    f.close()
    print("train.txt finash")
    # test_path = os.path.join(dataset_root, 'testing')
    # paths = os.listdir(test_path)
    # txt=''
    # for path in paths:
    #     txt += save_img(os.path.join(test_path,path), dataset_root)
    # f = open(os.path.join(txt_path,'val.txt'), 'w')
    # f.write(txt)
    # f.close()


