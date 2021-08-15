import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import grad

import time
import pyprind

from math import *

from model import *
from dataloader import *
from utils import *

##### Trainer #####

class Trainer:
    def __init__(self, DATA, CHECKPOINT,
                 generator_channels = [512, 512, 512, 512, 512, 256, 128, 64, 32, 16],
                 discriminator_channels = [512, 512, 512, 512, 512, 256, 128, 64, 32, 16],
                 style_dim = 512,
                 style_depth = 8,
                 lrs = {'128':0.0015, '256':0.002, '512':0.003, '1024':0.003},
                 betas = [0.0, 0.99],
                 batch_size = {'8':128, '16':64, '32':64, '64':32, '128':32, '256':16, '512':16, '1024':8}):
        self.DATA = DATA
        self.CHECKPOINT = CHECKPOINT
        self.generator_channels = generator_channels
        self.discriminator_channels = discriminator_channels
        self.style_dim = style_dim
        self.style_depth = style_depth
        self.lrs = lrs
        self.betas = betas
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dataset = CreateDataset(PATH=self.DATA)
        self.generator = Generator(channels=self.generator_channels, style_dim=self.style_dim, style_depth=self.style_depth).to(device)
        self.discriminator = Discriminator(channels=self.discriminator_channels).to(device)

        self.epochs = {'8':64, '16':64, '32':64, '64':128, '128':256, '256':256, '512':256, '1024':256}
    
    def grow(self):
        self.generator = self.generator.grow().to(device)
        self.discriminator = self.discriminator.grow().to(device)
        self.dataset = self.dataset.grow()
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size[str(self.dataset.image_size)], shuffle=True, drop_last=True)

        self.lr = self.lrs.get(str(self.dataset.image_size), 0.001)
        self.style_lr = self.lr * 0.01

        self.optimizer_d = optim.Adam(params=self.discriminator.parameters(), lr=self.lr, betas=self.betas)
        self.optimizer_g = optim.Adam([
                {'params': self.generator.model.parameters(), 'lr':self.lr},
                {'params': self.generator.style_mapper.parameters(), 'lr': self.style_lr},],
            betas=self.betas)
        
    def train_generator(self, batch_size, alpha):
        requires_grad(self.generator, True)
        requires_grad(self.discriminator, False)

        if random.random() < 0.9:
            z = [torch.randn(batch_size, self.style_dim).to(device),
                 torch.randn(batch_size, self.style_dim).to(device)]
        else:
            z = torch.randn(batch_size, self.style_dim).to(device)

        fake = self.generator(z, alpha=alpha)
        d_fake = self.discriminator(fake, alpha=alpha)
        loss = F.softplus(-d_fake).mean()
        loss.backward()

        self.optimizer_g.zero_grad()

        self.optimizer_g.step()

        return loss.item()

    def train_discriminator(self, real, batch_size, alpha):
        requires_grad(self.generator, False)
        requires_grad(self.discriminator, True)

        real.requires_grad = True
        self.optimizer_d.zero_grad()

        d_real = self.discriminator(real, alpha=alpha)
        loss_real = F.softplus(-d_real).mean()
        loss_real.backward(retain_graph=True)

        grad_real = grad(outputs=d_real.sum(), inputs=real, create_graph=True)[0]
        grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
        grad_penalty = 10 / 2 * grad_penalty
        grad_penalty.backward()

        if random.random() < 0.9:
            z = [torch.randn(batch_size, self.style_dim).to(device),
                 torch.randn(batch_size, self.style_dim).to(device)]
        else:
            z = torch.randn(batch_size, self.style_dim).to(device)

        fake = self.generator(z, alpha=alpha)
        d_fake = self.discriminator(fake, alpha=alpha)
        loss_fake = F.softplus(d_fake).mean()
        loss_fake.backward()

        loss = loss_real + loss_fake + grad_penalty
        self.optimizer_d.step()
        
        return loss.item(), (d_real.mean().item(), d_fake.mean().item())

    def run(self):
        flag, start_epoch = self.load_checkpoint()
        if flag:
            self.grow()

        self.generator.train()
        self.discriminator.train()

        while True:
            for epoch in range(start_epoch+1, self.epochs[str(self.dataset.image_size)]+1):
                print("Starting Epoch[{0}/{1}] | Image Size: {2}".format(epoch, self.epochs[str(self.dataset.image_size)], self.dataset.image_size))
                time.sleep(2)
                trained = 0
                epoch_loss_generator = 0
                epoch_loss_discriminator = 0

                bar = pyprind.ProgBar(len(self.dataloader), bar_char='█')
                for idx, batch in enumerate(self.dataloader, 1):
                    real = batch.to(device)
                    batch_size = batch.size(0)
                    trained += idx*batch_size
                    alpha = min(1, trained/len(self.dataset)) if self.dataset.image_size > 8 else 1

                    loss_d, (real_score, fake_score) = self.train_discriminator(real, real.size(0), alpha)
                    loss_g = self.train_generator(real.size(0), alpha)

                    epoch_loss_generator += loss_g/len(self.dataloader)
                    epoch_loss_discriminator += loss_d/len(self.dataloader)

                    bar.update()
                    torch.cuda.empty_cache()

                time.sleep(2)
                if epoch%2==0:
                    self.save_checkpoint(False, epoch)    
                print("Finished Epoch[{0}/{1}] | Image Size: {2} | Training Loss: Generator: {3} Discriminator: {4}".format(epoch, self.epochs[str(self.dataset.image_size)], self.dataset.image_size, epoch_loss_generator, epoch_loss_discriminator))

            start_epoch = 0
            self.save_checkpoint(True)     
            self.grow()
            if self.dataset.image_size > 1024:
                break

    def save_checkpoint(self, flag, epoch=0):
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'generator_optimizer': self.optimizer_g.state_dict(),
            'discriminator_optimizer': self.optimizer_d.state_dict(),
            'image_size': self.dataset.image_size,
            'flag': flag,
            'epoch': epoch,
        }, os.path.join(self.CHECKPOINT, "model.pth"))
        if flag:
            torch.save({
                'generator': self.generator.state_dict(),
                'discriminator': self.discriminator.state_dict(),
                'image_size': self.dataset.image_size,
            }, os.path.join(self.CHECKPOINT, "model-{}x{}.pth".format(self.dataset.image_size, self.dataset.image_size)))

    def load_checkpoint(self):
        if os.path.exists(os.path.join(self.CHECKPOINT, "model.pth")):
            checkpoint = torch.load(os.path.join(self.CHECKPOINT, "model.pth"))

            while self.dataset.image_size < checkpoint['image_size']:
                self.grow()
            
            self.generator.load_state_dict(checkpoint['generator'])
            self.discriminator.load_state_dict(checkpoint['discriminator'])
            self.optimizer_g.load_state_dict(checkpoint['generator_optimizer'])
            self.optimizer_d.load_state_dict(checkpoint['discriminator_optimizer'])
            flag = checkpoint.get('flag', True)
            start_epoch = checkpoint.get('epoch', 0)
        return flag, start_epoch

##### Inferencer #####

class Inferencer:
    def __init__(self, CHECKPOINT, 
                 generator_channels = [512, 512, 512, 512, 512, 256, 128, 64, 32, 16], 
                 style_dim = 512, 
                 style_depth = 8):
        self.CHECKPOINT = CHECKPOINT
        self.style_dim = style_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = 4
        self.generator = Generator(generator_channels, style_dim, style_depth).to(self.device)

        self.predictions = []

    def inference(self, n, image_size):
        test_z = torch.randn(n, self.style_dim).to(self.device)

        self.load_checkpoint(image_size)
        self.generator.eval()

        with torch.no_grad():            
            
            fake = self.generator(test_z, alpha=1)
            fake = (fake + 1) * 0.5
            fake = torch.clamp(fake, min=0.0, max=1.0)
            fake = F.interpolate(fake, size=(256, 256))
            fake = fake.detach().cpu().numpy()

            for index in range(n):
                self.predictions.append(np.moveaxis(fake[index], 0, -1)*255)

        return self.predictions

    def grow(self):
        self.generator = self.generator.grow().to(device)
        self.image_size *= 2
  
    def load_checkpoint(self, image_size):
        if os.path.exists(os.path.join(self.CHECKPOINT, "model-{}x{}.pth".format(image_size, image_size))):
            checkpoint = torch.load(os.path.join(self.CHECKPOINT, "model-{}x{}.pth".format(image_size, image_size)))

            while self.image_size < checkpoint['image_size']:
                self.grow()
            
            assert self.image_size == checkpoint['image_size']
            self.generator.load_state_dict(checkpoint['generator'])