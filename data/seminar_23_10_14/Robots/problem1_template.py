import numpy as np
import scipy.io


def read_file(name):
  mat_dict = scipy.io.loadmat(name)
  return [mat_dict["beacons"], mat_dict["robots"], mat_dict["x"], mat_dict["y"]]

beacons, robots, x, y = read_file("task1.mat")

A = np.zeros((robots.shape[0], max(robots) + max(beacons)))

'''
for i in xrange(A.shape[0]):
  A[ , ] =
  A[, ] =
'''

[res_x, _, _, _] = np.linalg.lstsq(A, x)
[res_y, _, _, _] = np.linalg.lstsq(A, y)

import matplotlib.pylab as plt

plt.plot(res_x[:max(robots) - 1], res_y[:max(robots) - 1], "r. ")
plt.plot(res_x[max(robots):], res_y[max(robots):], "b* ")

plt.show()
