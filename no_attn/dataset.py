#!/usr/bin/python
# encoding: utf-8
import os
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
# import lmdb
import six
import sys
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np

def extract_target(file_name):
    return file_name.split('_')[1].lower()

class WordImageDataset(Dataset):

    def __init__(self, root_dir, annotation_file, alphabet, transform=None,
                 max_samples=None, preload=False, downsample_factor=4, size=(100, 32)):
        """
        Args:
        """
        self.root_dir = root_dir
        self.path = []
        self.transform = transform

        with open(os.path.join(root_dir, annotation_file), "r") as f:
            for line in f:
                self.path.append(line.split(" ")[0])
        
        self.target = [extract_target(pth) for pth in self.path]
        self.max_len = max([len(t) for t in self.target]) + 1
        self.preload = preload
        self.size = size
        self.input_size = size[0] // downsample_factor - 2
        self.inv_alphabet = {word: i for i, word in enumerate(alphabet, 3)}
        # Store all the data in memory to speed up computations. HUGE MEMORY CONSUMPTION!!!
        if max_samples:
            idx = np.random.choice(len(self.path), max_samples, replace=False)
            self.path = [self.path[i] for i in idx]
            self.target = [self.target[i] for i in idx]
            
            if preload:
                self.images = [Image.open(os.path.join(root_dir, pth)).resize(self.size) for pth in self.path]


    def __getitem__(self, idx):
        if self.preload:
            img = self.images[idx]
        else:
            img = Image.open(os.path.join(self.root_dir, self.path[idx])).resize(self.size)
        
        if not self.transform is None:
            img = self.transform(img)

        return (img, self.target[idx])

    
    def __len__(self):
        return len(self.path)


    def get_max_len(self):
        return self.max_len
