# Autoencoder implementations

import torch
import torch.nn as nn
import torch.nn.functional as F

class Baseline(nn.Module):

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