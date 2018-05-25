# ------------------------------------------------------------ #
#
# file : dataAccessor.py
# author : CM
# Function to read, write and process the nii files from the
# dataset
#
# ------------------------------------------------------------ #

import os
from random import randint

import nibabel as nib
import numpy as np

# read nii file and load it into a numpy 3d array
def niiToNp(filename):
    data = nib.load(filename).get_data().astype('float32')
    return data/data.max()

# read a dataset and load it into a numpy 4d array
def readDataset(folder, size, size_x, size_y, size_z):
    dataset = np.empty((size, size_x, size_y, size_z))

    i = 0
    for filename in os.listdir(folder):
        if(i>=size):
            break
        dataset[i, :, :, :] = niiToNp(os.path.join(folder, filename))
        i = i+1

    return dataset

# reshape the dataset to match keras input shape (add channel dimension)
def reshapeDataset(d):
    return d.reshape(d.shape[0], d.shape[1], d.shape[2], d.shape[3], 1)

# exctract a patch from an image
def extractPatch(d, patch_size_x, patch_size_y, patch_size_z, x, y, z):
    patch = d[x:x+patch_size_x,y:y+patch_size_y,z:z+patch_size_z]
    return patch

# create random patchs for an image
def generateRandomPatchs(d, patch_size_x, patch_size_y, patch_size_z, patch_number):
    # max_patch_nb = (d.shape[0]-patch_size_x)*(d.shape[1]-patch_size_y)*(d.shape[2]-patch_size_z)
    data = np.empty((patch_number, patch_size_x, patch_size_y, patch_size_z))

    for i in range(0,patch_number):
        x = randint(0, d.shape[0]-patch_size_x)
        y = randint(0, d.shape[1]-patch_size_y)
        z = randint(0, d.shape[2]-patch_size_z)
        data[i] = extractPatch(d, patch_size_x, patch_size_y, patch_size_z, x, y, z)

    return data

# divide the full image into patchs
# todo : missing data if shape%patch_size is not 0
def generateFullPatchs(d, patch_size_x, patch_size_y, patch_size_z):
    patch_nb = int((d.shape[0]/patch_size_x)*(d.shape[1]/patch_size_y)*(d.shape[2]/patch_size_z))
    data = np.empty((patch_nb, patch_size_x, patch_size_y, patch_size_z))
    i = 0
    for x in range(0,d.shape[0], patch_size_x):
        for y in range(0, d.shape[1], patch_size_y):
            for z in range(0,d.shape[2], patch_size_z):
                data[i] = extractPatch(d, patch_size_x, patch_size_y, patch_size_z, x, y, z)
                i = i+1

    return data