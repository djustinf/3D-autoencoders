# Train Autoencoders

import os

from autoencoders import Baseline
from utils import MayaviDataset
import numpy as np
import mayavi.mlab
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

num_epochs = 100
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = MayaviDataset('./poles', transform=img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = Baseline(20*20*20, 128, 2, 4).cpu()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in dataloader:
        data = data.float()
        data = data.view(data.size(0), 8000)
        data = Variable(data).cpu()
        # ===================forward=====================
        output = model(data)
        loss = criterion(output, data)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, loss.data[0]))

torch.save(model.state_dict(), './3D_autoencoder.pth')