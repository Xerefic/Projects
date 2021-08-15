import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import grad

import os, glob
import sys
import time
import pyprind

import cv2
import matplotlib.pyplot as plt

import numpy as np
from math import *

from model import *
from dataloader import *
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "./checkpoints"
DATA = "./datasets/CelebA-HQ-img/"

##### Initializing the Inferencer #####

infrencer = Inferencer(CHECKPOINT=CHECKPOINT)

predictions = infrencer.inference(64, sys.argv[1])

##### Visualization #####

_, axis = plt.subplots(len(predictions)//4, 4, figsize=(16, len(predictions)))
axis = axis.flatten()
for image, ax in zip(predictions, axis):
    ax.imshow(image.astype('uint8'))
    ax.axis("off")
plt.show()