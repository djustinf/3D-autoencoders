# Train Autoencoders

import os

from autoencoders import Baseline
from autoencoders import ConvNet3D
from utils import MayaviDataset
from utils import Mnist3D
from utils import create_3D_noise
import numpy as np

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse

num_epochs = 50
batch_size = 128
learning_rate = 1e-3

parse = argparse.ArgumentParser()
parse.add_argument('-d', '--denoising' , action='store_true')
parse.add_argument('-o', '--outfile', required=True)
args = vars(parse.parse_args())

denoising = args['denoising']
outfile = args['outfile']

dataset = Mnist3D('./3D_mnist/full_dataset_vectors.h5')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = ConvNet3D().cpu()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
  for data in dataloader:
    data = data.float()

    if (denoising):
      corrupt_data = torch.from_numpy(create_3D_noise(data.numpy(), 0.15))
      corrupt_data = corrupt_data.view(corrupt_data.size(0), 4, 16, 16, 16)
      data = data.view(data.size(0), 4, 16, 16, 16)
      corrupt_data = Variable(corrupt_data).cpu()
      output = model(corrupt_data)
    else:
      data = data.view(data.size(0), 4, 16, 16, 16)
      data = Variable(data).cpu()
      output = model(data)

    loss = criterion(output, data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.data[0]))

torch.save(model.state_dict(), outfile)