import numpy as np
import torch
from torch.utils.data import Dataset
import os

class MayaviDataset(Dataset):

  def __init__(self, data_dir, transform=None):
    self.data_dir = data_dir 
    self.transform = transform

  def __len__(self):
    path, dirs, files = next(os.walk(self.data_dir))
    file_count = len(files)
    return file_count

  def __getitem__(self, idx):
    model = np.load(os.path.join(self.data_dir, '{}.npy'.format(str(idx))))

    if self.transform:
      model = self.transform(model)
    
    return model

def create_noise(img, fract):
  corrupt_image = img.copy()

  x_shape = img.shape[0]
  y_shape = img.shape[1]
  img_max = img.max()
  img_min = img.min()

  for i,x in enumerate(img):
    noise = np.random.randint(0, y_shape, int(fract*x_shape))
    for j in noise:
      corrupt_image[i][j] = img_max if (np.random.random() < 0.5) else img_min

  return corrupt_image