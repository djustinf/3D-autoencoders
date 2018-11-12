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

dataset = Mnist3D('./3D_mnist/full_dataset_vectors.h5', transform=img_transform)

model = Baseline(16*16*16, 128, 2, 4).cpu()
model.load_state_dict(torch.load('./3D_autoencoder.pth'))

img1 = dataset[0].float()
img1 = img1.view(1, 4096)
img1 = Variable(img1).cpu()
output1 = model.encode(img1)

img2 = dataset[0].float()
img2 = img2.view(1, 4096)
img2 = Variable(img2).cpu()
output2 = model.encode(img2)

interp = (0.5 * output1) + (0.5 * output2)
output = model.decode(interp)
output = output.view(16, 16, 16)

xx, yy, zz = np.where(output > 0)
xxyyzz = zip(xx, yy, zz)
for (xx, yy, zz) in xxyyzz:
  mayavi.mlab.points3d(xx, yy, zz,
                     mode="cube",
                     color=(1, 0, 0),
                     opacity=output[xx, yy, zz],
                     scale_factor=1)

mayavi.mlab.show()