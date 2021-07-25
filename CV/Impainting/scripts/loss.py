import torch
from torch import autograd

from utilies import *
from model import *

def calc_gradient_penalty(netD, real_data, fake_data, device):
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.to(device)

    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates = interpolates.requires_grad_().clone()

    disc_interpolates = netD(interpolates.float())
    grad_outputs = torch.ones(disc_interpolates.size())
    grad_outputs = grad_outputs.to(device)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                grad_outputs=grad_outputs, create_graph=True,
                                retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty