import torch
from torchvision import transforms
import os, glob
import cv2
import numpy as np

class CreateDataset(torch.utils.data.Dataset):
    def __init__(self, PATH, mode='training', n_classes=150):
        self.mode = mode
        self.n_classes = n_classes
        self.entry = np.array([os.path.splitext(os.path.basename(entry))[0] for entry in glob.glob(os.path.join(PATH, "images", self.mode, "*.jpg"))])
        np.random.shuffle(self.entry)
        if self.mode == 'training':
            max_size = 16384
        elif self.mode == 'validation':
            max_size = 128
        self.entry = self.entry[:max_size]
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def image_transform(self, image):
        image = np.float32(np.array(image))/255.
        image = image.transpose((2, 0, 1))
        image = self.normalize(torch.from_numpy(image.copy()))
        return image
    def label_transform(self, label):
        label = self.encode_label(label)
        label = label.transpose((2, 0, 1))
        return torch.from_numpy(label)

    def encode_label(self, label):
        encoded = np.zeros((label.shape[0], label.shape[1], self.n_classes))
        for row in range(label.shape[0]):
            for col in range(label.shape[1]):
                idx = label[row][col]-1
                encoded[row][col][idx] = 1
        return encoded

    def __getitem__(self, index):
        image = cv2.imread(os.path.join(PATH, "images", self.mode, str(self.entry[index]+ ".jpg")), 1)
        label = cv2.imread(os.path.join(PATH, "annotations", self.mode, str(self.entry[index]+ ".png")), 0)
        image = cv2.resize(image, (224,224), interpolation=cv2.INTER_AREA)
        label = cv2.resize(label, (224,224), interpolation=cv2.INTER_NEAREST)
        image = self.image_transform(image)
        label = self.label_transform(label)
        return {'image': image, 'label': label}

    def __len__(self):
        return len(self.entry)