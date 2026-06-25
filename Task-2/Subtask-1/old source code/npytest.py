from npy_append_array import NpyAppendArray
import numpy as np

a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])

with NpyAppendArray('test.npy', delete_if_exists=True) as npaa:
    npaa.append(a)
    npaa.append(b)

data = np.load('test.npy', mmap_mode='r')

print(data)