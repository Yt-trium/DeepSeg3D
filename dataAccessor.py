# ------------------------------------------------------------ #
#
# file : dataAccessor.py
# author : CM
# Function to read, write and process the nii files from the
# dataset
#
# ------------------------------------------------------------ #

import os
import nibabel as nib
import numpy as np

def niiToNp(filename):
    data = nib.load(filename).get_data().astype('float32')
    return data/data.max()

def readDataset(folder, size, size_x, size_y, size_z):
    dataset = np.empty((size, size_x, size_y, size_z))

    i = 0
    for filename in os.listdir(folder):
        if(i>=size):
            break
        dataset[i, :, :, :] = niiToNp(os.path.join(folder, filename))
        i = i+1

    return dataset