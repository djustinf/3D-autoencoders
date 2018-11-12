import numpy as np
from matplotlib.pyplot import cm
import h5py
import torch
from torch.utils.data import Dataset
import os

# Custom dataloader for custom Mayavi dataset we made
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

# Custom dataloader for a 3D mnist dataset we found on Kaggle
class Mnist3D(Dataset):

  def __init__(self, data_dir, transform=None):
    with h5py.File(data_dir, "r") as hf:    
      data = hf["X_train"][:]
    
    self.data_dir = np.zeros((data.shape[0], 4096, 3))

    for x in range(self.data_dir.shape[0]):
      self.data_dir[x] = add_extra_dims(data[x])
    
    self.data_dir = np.reshape(self.data_dir, (data.shape[0], 16, 16, 16, 3))
    self.transform = transform

  def __len__(self):
    return len(self.data_dir)

  def __getitem__(self, idx):
    model = self.data_dir[idx]
    model = np.swapaxes(model, 0, 3)

    if self.transform:
      model = self.transform(model)
    
    return torch.from_numpy(model)

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

def add_extra_dims(linear_model):
    color_map = cm.ScalarMappable(cmap="hsv")
    color_array = color_map.to_rgba(linear_model)[:,:-1]
    return color_array