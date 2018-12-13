import h5py
import argparse
from utils import create_3D_noise
import mayavi.mlab
import numpy as np

parse = argparse.ArgumentParser()
parse.add_argument('-f', '--file', type=str, required=True, help='File path')
parse.add_argument('-g', '--group', type=str, required=True, help='Group name')
parse.add_argument('-n', '--number', type=int, required=True, help='Model number')
args = vars(parse.parse_args())
  
model_file = args['file']
group = args['group']
num = args['number']

with h5py.File(model_file, "r") as hf:
  models = hf[group][:]
  data = models[num]
  for xx in range(16):
    for yy in range(16):
      for zz in range(16):
        color = data[xx, yy, zz]
        opacity = float(max(color[3], 0))
        color = (float(max(color[0], 0)), float(max(color[1], 0)), float(max(color[2], 0)))
        mayavi.mlab.points3d(xx, yy, zz,
                            mode="cube",
                            color=color,
                            opacity=opacity,
                            scale_factor=1)

  mayavi.mlab.show()