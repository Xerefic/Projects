import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import grad

import os, glob
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

##### Initializing the Trainer #####

trainer = Trainer(DATA=DATA, CHECKPOINT=CHECKPOINT)

##### Training #####

trainer.run()