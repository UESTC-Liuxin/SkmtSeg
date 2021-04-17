# -*- coding: utf-8 -*-
'''
@copyright: zwenc
@email: zwence@163.com
@Date: 2020-05-02 21:02:42
@FilePath: \SkmtSeg\predict.py
'''

import threading
import cv2
import torch
import numpy as np
import multiprocessing.dummy as mp
from modeling import build_skmtnet
import PIL.Image as Image
from tools.inference import Inferencer
import torchvision.transforms as T
from tools.postprocess import postprocess, decode_segmap
from UI.datainfo import DataInfo

def SegSkmt(self, image_path, seg):
    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        transform = T.Compose([
            T.Resize([512, 512]),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        img = Image.open(image_path)
        img = transform(img)
        img = torch.unsqueeze(img, dim=0).to(device)
        section = np.array([seg])
        input_s = {'image': img, 'section': section}
        pre = self.datainfo.infer.inference(input_s)
        post = postprocess(pre, 11)
        post = decode_segmap(post)

        self.datainfo.show_image = post

        cv2.imwrite("aaa.png", post)
        self.datainfo.message.append("{path} 处理完成 !!".format(path = image_path))
        # infer = Inferencer(self.model)
        # img = cv2.imread(image_path)
        # self.datainfo.img = transform(img).unsquzee(0).to(device)
        # self.datainfo.mask = infer.inference(self.datainfo.img)
    except Exception as e:
        print(e)

def loadModel(datainfo, modelPath):
    print("loading model {path}...........".format(path=modelPath))
    try:
        model = build_skmtnet(backbone='resnet50', auxiliary_head='fcn', trunk_head='deeplab_danet',
                            num_classes=11, output_stride=16)
                            
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        checkpoint = torch.load(modelPath)
        
        model.load_state_dict(checkpoint['model'])
        datainfo.infer = Inferencer(0, model)
        print("load success")
        datainfo.message.append("权重文件加载完成 !!")
        datainfo.message.append("load weight success")
        # return model
    except Exception as e:
        print(e)
        print("load weight failed !!")
        
class Predict():
    def __init__(self, config, callBack):
        self.mode = None
        self.datainfo = config
        self.callBack = callBack
        
        # init parametes
        self.process = None
        self.model = None
        self.result_image = None

        self.pool = mp.Pool(1)
        self.lock = mp.Lock()
        self.pool_max = 1

        self.loadmodel(self.callBack)
    
    def process_acquire(self):
        if self.pool_max < 1:
            return False
        try:
            self.lock.acquire()
            self.pool_max -= 1
        finally:
            self.lock.release()
        return True

    def process_release(self):
        try:
            self.lock.acquire()
            self.pool_max += 1
        finally:
            self.lock.release() 

    def run(self, imagePath, imageseg=None, callBack = None):
        self.callBack = callBack
        self.datainfo.mask = None
        self.datainfo.show_image = None

        if self.process_acquire() is False:
            self.datainfo.message.append("请等待上一个任务运行结束")
            print("请等待上一个任务运行结束")
            if self.callBack is not None:
                self.callBack()
            return
        
        ret = self.pool.apply_async(SegSkmt, (self, imagePath, imageseg), callback=self.process_callback)
        # ret.get()
        # self.process_release()
        # self.callBack()

    def loadmodel(self, callBack = None):
        self.callBack = callBack
        if self.process_acquire() is False:
            self.datainfo.message.append("请等待上一个任务运行结束")
            self.callBack()
            return
        model_path = self.datainfo.config["model"]["path"]
        result = self.pool.apply_async(loadModel, (self.datainfo, model_path), callback = self.process_callback)

        # result.get()

    def process_callback(self, msg):
        self.process_release()
        if self.callBack is not None:
            self.callBack()

    def show(self):
        if self.datainfo.show_image is not None:
            # cv2.imwrite("aaaa.png", self.datainfo.show_image)
            pass
            

