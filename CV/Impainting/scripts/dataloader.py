import torch
import torchvision
import cv2
import numpy as np
import os, glob


class CreateDataset(torch.utils.data.Dataset):
    def __init__(self, PATH, dataset, mode='train', sub_folder=True):
        self.PATH = PATH
        self.dataset = dataset
        self.mode = mode
        self.images = np.array([])
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if sub_folder:
            directories = self.find_sub_folders(os.path.join(self.PATH, self.dataset, self.mode))
            for directory in directories:
                entries = [ os.path.basename(entry) for entry in glob.glob(os.path.join(self.PATH, self.dataset, self.mode, directory, "*.jpg")) ]
                paths = [os.path.join(self.PATH, self.dataset, self.mode, directory, entry) for entry in entries]
                self.images = np.append(self.images, paths)
        else:
            entries = [ os.path.basename(entry) for entry in glob.glob(os.path.join(self.PATH, self.dataset, self.mode, "*.jpg")) ]
            paths = [os.path.join(self.PATH, self.dataset, self.mode, entry) for entry in entries]
            self.images = np.append(self.images, paths)

        np.random.shuffle(self.images)

    def find_sub_folders(self, directory):
        directories = [dir for dir in os.listdir(directory) if os.path.isdir(os.path.join(directory, dir))]
        return directories

    def image_transform(self, image):
        image = np.array(image)/255.
        image = image.transpose((2, 0, 1))
        image = self.normalize(torch.from_numpy(image.copy()))
        return image

    def __getitem__(self, index):
        image = cv2.imread(self.images[index], cv2.IMREAD_COLOR)
        image = cv2.resize(image, (256,256), interpolation=cv2.INTER_AREA)
        image = self.image_transform(image)

        return image
    
    def __len__(self):
        return len(self.images)