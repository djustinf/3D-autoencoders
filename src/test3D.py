import mayavi.mlab
import numpy

"""
file_num = 0
for i in range(19):
  for k in range(19):
    data = (20, 20, 20)
    data = numpy.zeros(data)
    data[0:20, i:i+1, k:k+1] = 1
    numpy.save('{}.npy'.format(file_num), data)
    file_num += 1

    data = (20, 20, 20)
    data = numpy.zeros(data)
    data[i:i+1, 0:20, k:k+1] = 1
    numpy.save('{}.npy'.format(file_num), data)
    file_num += 1

    data = (20, 20, 20)
    data = numpy.zeros(data)
    data[i:i+1, k:k+1, 0:20] = 1
    numpy.save('{}.npy'.format(file_num), data)
    file_num += 1
"""

data = (20, 20, 20)
data = numpy.zeros(data)
data[0:20, 0:1, 0:1] = 1

xx, yy, zz = numpy.where(data == 1)

mayavi.mlab.points3d(xx, yy, zz,
                     mode="cube",
                     color=(1, 0, 0),
                     scale_factor=1)

mayavi.mlab.show()