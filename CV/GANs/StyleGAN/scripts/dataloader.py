import torch
import torchvision.transforms as transforms

import os, glob

import cv2

import numpy as np


class CreateDataset(torch.utils.data.Dataset):
    def __init__(self, PATH, image_size=4):
        self.PATH = PATH
        self.image_size = image_size
        self.transform = self._get_transform_(self.image_size)
        self.entry = glob.glob(os.path.join(self.PATH, "*.jpg"))

    def _get_transform_(self, image_size):
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform

    def grow(self):
        self.image_size *= 2
        self.transform = self._get_transform_(self.image_size)

        return self

    def __getitem__(self, index):
        image = cv2.imread(self.entry[index], cv2.IMREAD_COLOR) 
        image = image.transpose((2, 0, 1))/255
        image = torch.from_numpy(image).float()
        image = self.transform(image)

        return image

    def __len__(self):
        return len(self.entry)