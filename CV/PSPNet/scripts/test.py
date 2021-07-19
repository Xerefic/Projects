import os, glob
import gc
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from resnet import resnet101, resnet50
from model import *
from dataloader import *

### Parameters ###

batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH = "/content/drive/MyDrive/Projects/Clubs/Analytics/Coord Projects/Model Zoo/PSPNet/datasets/ADEChallengeData2016"
CHECKPOINT = "/content/drive/MyDrive/Projects/Clubs/Analytics/Coord Projects/Model Zoo/PSPNet/checkpoints"

### Loading the Data ###

test_data = CreateDataset(PATH, mode='validation')

testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

### Initializing the Model ###

model = PSPNet()
model.to(device)

if os.path.exists(os.path.join(CHECKPOINT, "model.pth")):
    checkpoints = torch.load(os.path.join(CHECKPOINT, "model.pth"))

    model.load_state_dict(checkpoints['model_state_dict'])

### Testing ###

predictions = []

with torch.no_grad():
    for idx, batch in enumerate(valloader, 1):
        image = batch['image'].to(device)
        label = batch['label'].to(device)
        preds = model(image)
        preds_ = preds.detach().cpu().numpy()
        label_ = label.detach().cpu().numpy()
        image_ = image.detach().cpu().numpy()
        
        print("Batch[{}]".format(idx))

        for batch in range(preds.shape[0]):
            decoded = np.zeros((224,224))
            decode = preds_[batch]
            decode = np.moveaxis(decode, 0, -1)
            encoded = np.zeros((224,224))
            encode = label_[batch]
            encode = np.moveaxis(encode, 0, -1)
            actual = image_[batch]
            actual = np.moveaxis(actual, 0, -1)
            for row in range(decoded.shape[0]):
                for col in range(decoded.shape[1]):
                    decoded[row][col] = np.argmax(decode[row][col])+1
                    encoded[row][col] = np.argmax(encode[row][col])+1
            predictions.append({'image': actual, 'label': encoded, 'predicted': decoded})

def show(predictions, index):
    image = predictions[index]['image']*255
    label = predictions[index]['label']
    predicted = predictions[index]['predicted']

    return image, label, predicted

for index in range(len(predictions)):
    image, label, predicted = show(predictions, index)
    cv2_imshow(image)
    cv2.waitKey(0)
    cv2_imshow(np.column_stack([label, predicted]))
    cv2.waitKey(0)
