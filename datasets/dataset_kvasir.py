import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from einops import repeat
from icecream import ic
from PIL import Image
import sys
import csv
import torchvision.transforms as transforms


class RandomGenerator(object):
    def __init__(self, output_size, low_res, split=None):
        self.output_size = output_size
        self.low_res = low_res
        self.split = split

    def __call__(self, sample):
        image, label, image_path = sample['image'], sample['label'], sample['path']
        
        x, y, _ = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y, 1), order=0)
        
        label_h, label_w, c = label.shape
        low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w, 1), order=0)
        image = torch.from_numpy(image.astype(np.float32)).permute([2,0,1])
        label = torch.from_numpy(label.astype(np.float32))[:,:,0]
        low_res_label = torch.from_numpy(low_res_label.astype(np.float32))[:,:,0]
        
        sample = {'image': image, 'label': label.long(), 'low_res_label': low_res_label.long(), 'path': image_path}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, train_dir, num_data=0, transform=None, dataset=None):
        self.transform = transform  # using transform in torch!
        self.train_file = train_dir
        self.num_data = num_data
        
        if self.num_data > 0:
            self.sample_list = os.listdir(self.train_file)[:self.num_data]
        else:
            self.sample_list = os.listdir(self.train_file)
        

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.train_file, self.sample_list[idx])
        image = Image.open(image_path).convert('RGB')
        label = Image.open(image_path.replace('/Images', '/Masks')).convert('RGB')
        image, label = np.array(image) / 255.0, np.uint8(np.array(label) > 0)
        
        # annot = self.load_annotations(idx)
        # Input dim should be consistent
        # Since the channel dimension of nature image is 3, that of medical image should also be 3
        sample = {'image': image, 'label': label, 'path': image_path}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        # sample['anno'] = annot[0][np.newaxis, :]
        return sample

