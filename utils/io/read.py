# ------------------------------------------------------------ #
#
# file : utils/io/read.py
# author : CM
# Function to read dataset
#
# ------------------------------------------------------------ #

import os
import sys

import nibabel as nib
import numpy as np

# read nii file and load it into a numpy 3d array
def niiToNp(filename):
    data = nib.load(filename).get_data().astype('float16')
    return data/data.max()

# read a dataset and load it into a numpy 4d array
def readDataset(folder, size, size_x, size_y, size_z):
    dataset = np.empty((size, size_x, size_y, size_z), dtype='float16')
    i = 0
    files = os.listdir(folder)
    files.sort()
    for filename in files:
        if(i>=size):
            break
        print(filename)
        dataset[i, :, :, :] = niiToNp(os.path.join(folder, filename))
        i = i+1

    return dataset

# reshape the dataset to match keras input shape (add channel dimension)
def reshapeDataset(d):
    return d.reshape(d.shape[0], d.shape[1], d.shape[2], d.shape[3], 1)

# read a dataset and load it into a numpy 3d array as raw data (no normalisation)
def readRawDataset(folder, size, size_x, size_y, size_z, dtype):
    files = os.listdir(folder)
    files.sort()

    if(len(files) < size):
        sys.exit(2)

    count = 0
    # astype depend on your dataset type.
    dataset = np.empty((size, size_x, size_y, size_z)).astype(dtype)

    for filename in files:
        if(count>=size):
            break
        dataset[count, :, :, :] = nib.load(os.path.join(folder, filename)).get_data()
        count += 1
        print(count, '/', size, os.path.join(folder, filename))

    return dataset