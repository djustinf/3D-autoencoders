from utils import add_extra_dims
import argparse
import random
import math
import h5py
import numpy as np

def distance(pt1, pt2):
  return math.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2 + (pt2[2]-pt1[2])**2)

def gen_sphere(x, y, z, radius):
  sphere = np.zeros((16, 16, 16))
  for i in range(16):
    for j in range(16):
      for k in range(16):
        if distance((x, y, z), (i, j, k)) <= radius:
          sphere[i, j, k] = 1

  return sphere

parse = argparse.ArgumentParser()
parse.add_argument('-n', '--number', type=int, required=True, help='Number of spheres')
args = vars(parse.parse_args())

num = args['number']

models = []
for x in range(num):
  xx = random.randint(0, 15)
  yy = random.randint(0, 15)
  zz = random.randint(0, 15)
  min_dim = min(xx, yy, zz)
  max_dim = max(xx, yy, zz)
  # generate random radius that fits in the 16x16x16 box
  r = min(min_dim, 15-max_dim)
  sphere = np.reshape(gen_sphere(xx, yy, zz, r), (4096))
  models.append(sphere)

new_models = np.vstack(models)
color_models = np.zeros((new_models.shape[0], 4096, 4))
for x in range(new_models.shape[0]):
  color_models[x] = add_extra_dims(new_models[x])
    
color_models = np.reshape(color_models, (new_models.shape[0], 16, 16, 16, 4))
with h5py.File('./spheres.h5', 'w') as hf:
  hf['spheres'] = color_models