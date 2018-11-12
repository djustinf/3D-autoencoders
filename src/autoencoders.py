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

  def __init__(self):
    super(ConvNet3D, self).__init__()

    encoder_layers = []
    encoder_layers.append(nn.Conv3d(16, 14, 3, stride=1))
    encoder_layers.append(nn.ReLU(True))
    encoder_layers.append(nn.MaxPool3d(2, stride=2, padding=1)) # 8x8x8
    encoder_layers.append(nn.Conv3d(8, 6, 3, stride=1))
    encoder_layers.append(nn.ReLU(True))
    encoder_layers.append(nn.MaxPool3d(2, stride=2, padding=1)) # 4x4x4
    encoder_layers.append(nn.Conv3d(4, 2, 3, stride=1))

    self.encoder = nn.Sequential(*encoder_layers)

    decoder_layers = []

    decoder_layers.append(nn.ConvTranspose3d(2, 4, 3, stride=1))
    decoder_layers.append(nn.MaxUnpool3d(2, stride=2, padding=1)) # 6x6x6
    decoder_layers.append(nn.ReLU(True))
    decoder_layers.append(nn.ConvTranspose3d(6, 8, 3, stride=1))
    decoder_layers.append(nn.MaxUnpool3d(2, stride=2, padding=1)) # 14x14x14
    decoder_layers.append(nn.ReLU(True))
    decoder_layers.append(nn.ConvTranspose3d(14, 16, 3, stride=1))

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