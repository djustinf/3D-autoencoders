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

for xx in range(16):
    for yy in range(16):
        for zz in range(16):
            color = output[xx, yy, zz]
            opacity = max(color[3].item(), 0)
            color = (max(color[0].item(), 0), max(color[1].item(), 0), max(color[2].item(), 0))
            mayavi.mlab.points3d(xx, yy, zz,
                                mode="cube",
                                color=color,
                                opacity=opacity,
                                scale_factor=1)

mayavi.mlab.show()