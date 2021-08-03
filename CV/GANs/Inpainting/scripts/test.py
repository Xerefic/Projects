import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary
from torch import autograd
import gc

import time
import pyprind

import matplotlib.pyplot as plt

from model import *
from dataloader import *
from loss import *
from utilies import *

### Parameters ###

batch_size = 6
start_epochs = 0
total_epochs = 8 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "/content/drive/MyDrive/Projects/Clubs/Analytics/Coord Projects/Model Zoo/Impainting/checkpoints"
PATH = "/content/drive/MyDrive/Projects/Clubs/Analytics/Coord Projects/Model Zoo/Inpainting/datasets"

### Loading the Data ###

test_data = CreateDataset(PATH, "imagenet12", mode='test', sub_folder=False)

testloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

### Initializing the Model ###

net_gen = Generator(use_cuda=True)

net_gen.to(device)

if os.path.exists(os.path.join(CHECKPOINT, "net_gen.pth")):
    checkpoints = torch.load(os.path.join(CHECKPOINT, "net_gen.pth"))

    net_gen.load_state_dict(checkpoints['net_gen_state_dict'])

### Testing ### 

preds = []

for idx, ground_truth in enumerate(testloader, 1):
	batch_size = ground_truth.size(0)
    bboxes = random_bbox(batch_size=batch_size)
    x, mask = mask_image(ground_truth, bboxes)

    ground_truth = ground_truth.to(device)
    x = x.to(device)
    mask = mask.to(device)

    x1, x2, offset_flow = net_gen(x.float(), mask)
    x2_inpaint = x2 * mask + x * (1. - mask)

    for index in range(batch_size):
        ground = ground_truth[index].detach().cpu().numpy()
        masked = x[index].detach().cpu().numpy()
        image = x2_inpaint[index].detach().cpu().numpy()
        
        ground = np.moveaxis(ground, 0, -1)*255
        masked = np.moveaxis(masked, 0, -1)*255
        image = np.moveaxis(image, 0, -1)*255

        preds.append({'ground':ground, 'masked':masked, 'image':image})

### Visualization ###

for index in range(len(preds)):
    render = preds[index]
    cv2.imshow(np.concatenate((render['ground'], render['masked'], render['image']), axis=1))