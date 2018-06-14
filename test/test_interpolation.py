# ------------------------------------------------------------ #
#
# file : test/test_interpolation.py
# author : CM
# Tests for interpolation functions
#
# ------------------------------------------------------------ #

from scipy.ndimage import zoom, rotate
import numpy as np
import time

data = np.random.rand(4,4,4)
dataR = rotate(input=data, angle=180, axes=(1,0), reshape=False)
dataZ = zoom(input=data, zoom=0.5)

print(data)
print(dataR)
print(dataZ)

print(data.shape)
print(dataR.shape)
print(dataZ.shape)

data = np.random.rand(100,16,16,16,1)

start = time.time()
for i in data:
    dataR = rotate(input=i, angle=30, axes=(1,0), reshape=False)
end = time.time()

print(end - start)
print(dataR.shape)

start = time.time()
for i in data:
    dataZ = zoom(input=i, zoom=2)
end = time.time()

print(end - start)
print(dataZ.shape)

start = time.time()
for i in data:
    dataZ = zoom(input=i, zoom=0.5)
end = time.time()

print(end - start)
print(dataZ.shape)

start = time.time()
data = np.random.rand(448,448,224)
dataZ = zoom(input=data, zoom=2)
end = time.time()

print(end - start)
print(dataZ.shape)