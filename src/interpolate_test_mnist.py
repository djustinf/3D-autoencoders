# Perform Linear Interpolation 

import os

from autoencoders import Baseline
import torch
import cv2
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = MNIST('./data', transform=img_transform, download=True)

model = Baseline(28*28, 128, 2, 4).cpu()
model.load_state_dict(torch.load('./sim_autoencoder.pth'))

img1, _ = dataset[14]
img1 = img1.view(img1.size(0), -1)
save_image(to_img(img1), './mlp_img/test_interpolate_5.png')
img1 = Variable(img1).cpu()
output1 = model.encode(img1)

img2, _ = dataset[15]
img2 = img2.view(img2.size(0), -1)
save_image(to_img(img2), './mlp_img/test_interpolate_6.png')
img2 = Variable(img2).cpu()
output2 = model.encode(img2)

interp = (0.5 * output1) + (0.5 * output2)
output = model.decode(interp)
save_image(to_img(output), './mlp_img/test_interpolate_interp_56.png')
