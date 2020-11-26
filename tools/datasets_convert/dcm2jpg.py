#coding=utf-8
import os
import cv2
# import matplotlib.pylab as plt
import numpy as np
import pydicom
from pydicom import dcmread


def convert_ybr_to_rgb(arr):
    if len(arr.shape) == 4:
        return np.vstack([convert_ybr_to_rgb(a)[np.newaxis] for a in arr])
    else:
        temp = arr[..., 1].copy()
        arr[..., 1] = arr[..., 2]
        arr[..., 2] = temp
        return cv2.cvtColor(arr, cv2.COLOR_YCR_CB2RGB)


def get_pixel_array_rgb(ds):
    if ds.PhotometricInterpretation in ['YBR_FULL', 'YBR_FULL_422']:
        return convert_ybr_to_rgb(ds.pixel_array)

    return ds.pixel_array


def read_information(ds):
    for col in ds:
        print(col)

def dcm2jpg(data_root):
    dcm_root=os.path.join(data_root,'DICOM','S63160','S4010')
    for file_index,file_name in enumerate(os.listdir(dcm_root)):
        file_path=os.path.join(data_root,'JPEG',file_name)
        #create sub dirs
        isExists = os.path.exists(file_path)
        if not isExists:
            print('make_dir')
            os.makedirs(file_path)
        # read all dcom files
        ds=dcmread(os.path.join(dcm_root,file_name))
        # print(ds.PhotometricInterpretation)
        # read_information(ds)
        imgs=get_pixel_array_rgb(ds)
        if imgs.ndim >2:
            for img_index in range(imgs.shape[0]):
                img_name=os.path.join(file_path,'20_9_5'+file_name+'_'+str(img_index)+'.jpg')
                try:
                    img=cv2.cvtColor(imgs[img_index],cv2.COLOR_GRAY2RGB)
                except:
                    img=imgs[img_index]
                cv2.imwrite(img_name,img)
        else:
            img_name = os.path.join(file_path, '20_9_5'+file_name + '_' + '.jpg')
            cv2.imwrite(img_name, imgs)





def rgbtojpg():
    pass



if __name__ =='__main__':
    data_root=os.path.join('F:\\CV\\Project\\SKMT\\mmsegmentation\\data')
    dcm2jpg(data_root)

