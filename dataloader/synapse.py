"""
@author: Maglan
@contact: maglanyulan@163.com
@Created on: DATE{TIME}
"""
import os
import random
import h5py
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torchvision import transforms
from dataloader.transforms_utils import custom_transforms as tr
from mypath import Path


class Synapse_dataset(Dataset):
    CLASSES = ('background', '1', '2', '3',
               '4', '5', '6', '7', '8')
    PALETTE = np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
                          [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],])
    CLASSES_PIXS_WEIGHTS=(0.7450,0.0501,0.0016,0.0932 ,0.0611 ,
                          0.0085,0.0092,0.0014,0.0073,0.0012,0.0213)

    NUM_CLASSES = 9
    def __init__(self, args,base_dir=Path.db_root_dir('synapse'),split='train'):

        self._base_dir = base_dir
        self.args = args
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')
        self._cat_dir = os.path.join(self._base_dir, 'SegmentationClass')
        _splits_dir = os.path.join(self._base_dir, 'ImageSets')

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.im_ids = []
        self.images = []
        self.categories = []


        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image = os.path.join(self._image_dir, line + ".jpg")
                _cat = os.path.join(self._cat_dir, line + ".png")
                # print(_image)
                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)

        assert (len(self.images) == len(self.categories))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = Image.open(self.categories[idx])

        sample = {'image': image, 'label': label}
        for split in self.split:
            if split == "train":
                return self.transform_tr(sample)
            elif split == 'val':
                return self.transform_val(sample)



    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            #tr.RandomHorizontalFlip(),
            # tr.RandomScaleCrop(base_size=self.args.image_size, crop_size=self.args.crop_size),
            tr.FixedResize(self.args.crop_size),
            #tr.RandomGaussianBlur(),
            #resnet
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            #tr.Normalize(mean=(0.318, 0.318, 0.316), std=(0.114, 0.114, 0.115)),
            tr.ToTensor()]
            )

        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

        # get_ISPRS and encode_segmap generate label map[
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
        return 'synapse(split=' + str(self.sample_list) + ')'
