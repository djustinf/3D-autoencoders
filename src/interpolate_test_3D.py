# Perform Linear Interpolation 

import os

from autoencoders import Baseline
from autoencoders import ConvNet3D
from utils import MayaviDataset
from utils import Mnist3D
import mayavi.mlab
import numpy as np
import argparse
import torch
import cv2
import torchvision
import h5py
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

parse = argparse.ArgumentParser()
parse.add_argument('-n', '--number', type=int, required=True, help='Number of models to generate')
args = vars(parse.parse_args())
  
num = args['number']

dataset = Mnist3D('./3D_mnist/full_dataset_vectors.h5')

model = ConvNet3D().cpu()
model.load_state_dict(torch.load('./3D_autoencoder.pth'))

img1 = dataset[0].float()
img1 = img1.view(1, 4, 16, 16, 16)
img1 = Variable(img1).cpu()
output1 = model.encode(img1)

img2 = dataset[3].float()
img2 = img2.view(1, 4, 16, 16, 16)
img2 = Variable(img2).cpu()
output2 = model.encode(img2)

models = []
for x in range(num):
  cur = 1.0/num * x
  interp = (cur * output1) + ((1-cur) * output2)
  output = model.decode(interp)
  output = output.view(4, 16, 16, 16)
  output = np.swapaxes(output.numpy(), 0, 3)
  models.append(output)

new_models = np.vstack(models)
with h5py.File('./models.h5', 'w') as hf:
  hf['models'] = new_models
