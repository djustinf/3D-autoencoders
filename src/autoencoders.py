# Autoencoder implementations

from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoBase(nn.Module):
  __metaclass__ = ABCMeta

  @abstractmethod
  def forward(self, val):
    pass

  @abstractmethod
  def encode(self, val):
    pass

  @abstractmethod
  def decode(self, val):
    pass

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

# Still broken for now, padding/strides need to be added to make the math work out correctly
class ConvNet3D(AutoBase):

  def __init__(self, dimensions, start, scale, depth):
    super(ConvNet3D, self).__init__()

    encoder_layers = []

    cur_dim = start
    encoder_layers.append(nn.Conv3d(dimensions, cur_dim, 3))
    encoder_layers.append(nn.ReLU(True))
    encoder_layers.append(nn.MaxPool3d(2))
    for x in range(depth):
      encoder_layers.append(nn.Conv3d(cur_dim, int(cur_dim/scale), 3))
      encoder_layers.append(nn.ReLU(True))
      encoder_layers.append(nn.MaxPool3d(2))
      cur_dim = int(cur_dim/scale)

    self.encoder = nn.Sequential(*encoder_layers)

    decoder_layers = []

    for x in range(depth-1):
      decoder_layers.append(nn.ConvTranspose3d(cur_dim, int(cur_dim*scale), 3))
      decoder_layers.append(nn.ReLU(True))
      cur_dim = int(cur_dim*scale)

    decoder_layers.append(nn.ConvTranspose3d(cur_dim, dimensions, 3))
    decoder_layers.append(nn.Tanh()) # figure out why the tanh matters here

    self.decoder = nn.Sequential(*decoder_layers)

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