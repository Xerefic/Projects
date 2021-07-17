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

### Parameters ###

batch_size = 16
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "/content/drive/MyDrive/Projects/Clubs/Analytics/Coord Projects/Model Zoo/PSPNet/checkpoints"
PATH = "/content/drive/MyDrive/Projects/Clubs/Analytics/Coord Projects/Model Zoo/PSPNet/datasets/ADEChallengeData2016"

### Loading the Data ###

train_data = CreateDataset(PATH, mode='training')
val_data = CreateDataset(PATH, mode='validation')

trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

### Initializing the Model ###

model = PSPNet()
model.to(device)

criterion = nn.BCELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.5, 0.999))

### Training ###

train_loss = []
train_acc = []
val_loss = []
val_acc = []

def plot_loss(epoch, train_loss, val_loss):
    epochs = np.arange(epoch+1)
    plt.plot(epochs, train_loss, label="Train Loss", color="green")
    plt.plot(epochs, val_loss, label="Validation Loss", color="red")

    plt.title("Training Loss - "+str(epoch))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="best")

    plt.savefig(os.path.join(CHECKPOINT, "Loss.png"))
    plt.close()

for epoch in range(epochs):

    gc.collect()
    torch.cuda.empty_cache()

    model.train()
    epoch_train_loss = []
    epoch_val_loss = []
    for idx, batch in enumerate(trainloader, 1):
        image = batch['image'].to(device)
        label = batch['label'].to(device)
        preds = model(image)
        loss = criterion(preds.float(), label.float())
        
        loss.backward()
        optimizer.step()
        epoch_train_loss.append(loss.item())
        print("Epoch[{0}]: Batch[{1}]   Train Loss:{2}".format(epoch+1, idx, loss.item()))
        gc.collect()
        torch.cuda.empty_cache()

    train_loss.append(epoch_train_loss[-1])

    gc.collect()
    torch.cuda.empty_cache()

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(valloader, 1):
            image = batch['image'].to(device)
            label = batch['label'].to(device)
            preds = model(image)
            loss = criterion(preds.float(), label.float())

            epoch_val_loss.append(loss.item())
            print("Epoch[{0}]: Batch[{1}]   Val Loss:{2}".format(epoch+1, idx, loss.item()))

        val_loss.append(epoch_val_loss[-1])

    gc.collect()
    torch.cuda.empty_cache()

    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss[-1],
            }, os.path.join(PATH, "model.pth"))
    plot_loss(epoch, train_loss, val_loss)