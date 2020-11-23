# -*- coding: utf-8 -*-
"""
@author: LiuXin
@contact: xinliu1996@163.com
@Created on: DATE{TIME}
"""
from __future__ import print_function, division
import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloader import custom_transforms as tr


class ToTensor():
    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        section =sample['section']
        
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)
        section=np.array(section).astype(np.long)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()
        section=torch.from_numpy(section).long()

        return {'image': img,
                'label': mask,
                'section':section}





class SkmtDataSet(Dataset):
    """
    PascalVoc dataset
    """
    CLASSES = ('background', 'SAS', 'LHB', 'D',
               'HH', 'SUB', 'SUP', 'GL', 'GC',
               'SCB', 'INF', 'C', 'TM', 'SHB',
               'LHT', 'SAC', 'INS','BBLH','LHBT')

    PALETTE = np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
               [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
               [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
               [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],[255,255,255],[255,255,255]])

    CLASSES_PIXS_WEIGHTS=(0.7450,0.0501,0.0016,0.0932 ,0.0611 ,
                          0.0085,0.0092,0.0014,0.0073,0.0012,0.0213,
                          0,0,0,0,0,0,0,0)

    NUM_CLASSES = len(CLASSES)

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('skmt'),
                 split='train',
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')
        self._cat_dir = os.path.join(self._base_dir, 'SegmentationClass')

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args

        _splits_dir = os.path.join(self._base_dir, 'ImageSets')

        self.im_ids = []
        self.images = []
        self.categories = []

        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image = os.path.join(self._image_dir, line + ".jpg")
                _cat = os.path.join(self._cat_dir, line + ".png")
                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        _section = self._get_section(index)-1

        sample = {'image': _img, 'label': _target,'section':_section}
        for split in self.split:
            if split == "train":
                return self.transform_tr(sample)
            elif split == 'val':
                return self.transform_val(sample),self.images[index].split('/')[-1]

    def _get_section(self,index):
        _name=self.images[index].split('/')[-1]
        _section=_name.split('_')[0][-2]
        return int(_section)

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.categories[index])

        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.image_size, crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()])

        return composed_transforms(sample)

        # get_ISPRS and encode_segmap generate label map


    @classmethod
    def encode_segmap(cls, mask):
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

        for ii, label in enumerate(cls.PALETTE):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask

    @classmethod
    def decode_segmap(cls, label_mask):
        """Decode segmentation class labels into a color image
        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
              in a figure.
        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        label_colours = cls.PALETTE
        n_classes = len(label_colours)

        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def __str__(self):
        return 'skmt(split=' + str(self.split) + ')'






