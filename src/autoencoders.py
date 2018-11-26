# Autoencoder implementations

from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoBase(nn.Module):
  __metaclass__ = ABCMeta

  @abstractmethod
  def __init__(self):
    super(AutoBase, self).__init__()

  def forward(self, val):
    val = self.encoder(val)
    val = self.decoder(val)
    return val

  def encode(self, val):
    val = self.encoder(val)
    return val.cpu().data

  def decode(self, val):
    val = self.decoder(val)
    return val.cpu().data

class Baseline(AutoBase):

  def __init__(self, dimensions, start, scale, depth):
    super(Baseline, self).__init__()

    encoder_layers = []

    cur_dim = start
    encoder_layers.append(nn.Linear(dimensions, cur_dim))
    for x in range(depth):
      encoder_layers.append(nn.ReLU(True))
      encoder_layers.append(nn.Linear(cur_dim, int(cur_dim/scale)))
      cur_dim = int(cur_dim/scale)

    self.encoder = nn.Sequential(*encoder_layers)

    decoder_layers = []

    for x in range(depth-1):
      decoder_layers.append(nn.Linear(cur_dim, int(cur_dim*scale)))
      decoder_layers.append(nn.ReLU(True))
      cur_dim = int(cur_dim*scale)

    decoder_layers.append(nn.Linear(cur_dim, dimensions))
    decoder_layers.append(nn.Tanh()) # figure out why the tanh matters here

    self.decoder = nn.Sequential(*decoder_layers)

class ConvNet3D(AutoBase):

  def __init__(self):
    super(ConvNet3D, self).__init__()

    encoder_layers = []
    encoder_layers.append(nn.Conv3d(4, 64, 3, stride=1))
    encoder_layers.append(nn.ReLU(True))
    encoder_layers.append(nn.MaxPool3d(2, stride=2))
    encoder_layers.append(nn.Conv3d(64, 32, 3, stride=1))
    encoder_layers.append(nn.ReLU(True))
    encoder_layers.append(nn.MaxPool3d(2, stride=2))

    self.encoder = nn.Sequential(*encoder_layers)

    decoder_layers = []
    decoder_layers.append(nn.ConvTranspose3d(32, 64, 5, stride=2))
    decoder_layers.append(nn.ReLU(True))
    decoder_layers.append(nn.ConvTranspose3d(64, 32, 3, stride=2))
    decoder_layers.append(nn.ReLU(True))
    decoder_layers.append(nn.ConvTranspose3d(32, 4, 2, stride=1))
    decoder_layers.append(nn.ReLU(True))

    decoder_layers.append(nn.Tanh()) # figure out why the tanh matters here

    self.decoder = nn.Sequential(*decoder_layers)