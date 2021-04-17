# -*- coding: utf-8 -*-
r'''
@copyright: zwenc
@email: zwence@163.com
@Date: 2021-04-11 17:22:02
@FilePath: \SkmtSeg\UI\datainfo.py
'''

import configparser

class DataInfo(object):

    def __init__(self):
        pass
    
    def init(self, config_path):
        self.load_config(config_path)

        self.mask = None
        self.img = None

        self.show_image = None
        self.infer = None

        self.message = []
    
    def load_config(self, config_path):

        con = configparser.ConfigParser()
        con.read(config_path, encoding='utf-8')
        self.config = con

    @classmethod
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super().__new__(cls)
        return cls._instance


