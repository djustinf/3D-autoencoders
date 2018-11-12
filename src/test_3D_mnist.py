import h5py
import mayavi.mlab
import numpy

with h5py.File("./3D_mnist/full_dataset_vectors.h5", "r") as hf:    
  X_train = hf["X_train"][:]
  y_train = hf["y_train"][:]
  X_test = hf["X_test"][:]
  y_test = hf["y_test"][:]

data = X_train[0]
data = numpy.reshape(data, (16, 16, 16))
xx, yy, zz = numpy.where(data > 0)
xxyyzz = zip(xx, yy, zz)
for (xx, yy, zz) in xxyyzz:
  mayavi.mlab.points3d(xx, yy, zz,
                     mode="cube",
                     color=(1, 0, 0),
                     opacity=data[xx, yy, zz],
                     scale_factor=1)

mayavi.mlab.show()