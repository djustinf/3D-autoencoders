# Perform Linear Interpolation 

import os

from autoencoders import Baseline
from utils import MayaviDataset
from utils import Mnist3D
import mayavi.mlab
import numpy as np
import torch
import cv2
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

img_transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = Mnist3D('./3D_mnist/full_dataset_vectors.h5')

model = Baseline(4096*4, 64, 2, 2).cpu()
model.load_state_dict(torch.load('./3D_autoencoder.pth'))

img1 = dataset[0].float()
img1 = img1.view(1, 4096*4)
img1 = Variable(img1).cpu()
output1 = model.encode(img1)

img2 = dataset[1].float()
img2 = img2.view(1, 4096*4)
img2 = Variable(img2).cpu()
output2 = model.encode(img2)

interp = (0.5 * output1) + (0.5 * output2)
output = model.decode(interp)
output = output.view(4, 16, 16, 16)
output = torch.from_numpy(np.swapaxes(output.numpy(), 0, 3))

for xx in range(16):
    for yy in range(16):
        for zz in range(16):
            color = output[xx, yy, zz]
            opacity = max(color[3].item(), 0)
            color = (max(color[0].item(), 0), max(color[1].item(), 0), max(color[2].item(), 0))
            mayavi.mlab.points3d(xx, yy, zz,
                                mode="cube",
                                color=color,
                                opacity=opacity,
                                scale_factor=1)

mayavi.mlab.show()