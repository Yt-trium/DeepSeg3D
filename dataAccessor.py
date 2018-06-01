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

# ----- Input -----
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

# ----- Output -----
# write nii file from a numpy 3d array
def npToNii(data, filename):
    axes = np.eye(4)
    axes[0][0] = -1
    axes[1][1] = -1
    image = nib.Nifti1Image(data, axes)
    nib.save(image, filename)

# ----- Patch Extraction -----
# -- Single Patch
# exctract a patch from an image
def extractPatch(d, patch_size_x, patch_size_y, patch_size_z, x, y, z):
    patch = d[x:x+patch_size_x,y:y+patch_size_y,z:z+patch_size_z]
    return patch

# create random patch for an image
def generateRandomPatch(d, patch_size_x, patch_size_y, patch_size_z):
    x = randint(0, d.shape[0]-patch_size_x)
    y = randint(0, d.shape[1]-patch_size_y)
    z = randint(0, d.shape[2]-patch_size_z)
    data = extractPatch(d, patch_size_x, patch_size_y, patch_size_z, x, y, z)
    return data

# -- Multiple Patchs
# create random patchs for an image
def generateRandomPatchs(d, patch_size_x, patch_size_y, patch_size_z, patch_number):
    # max_patch_nb = (d.shape[0]-patch_size_x)*(d.shape[1]-patch_size_y)*(d.shape[2]-patch_size_z)
    data = np.empty((patch_number, patch_size_x, patch_size_y, patch_size_z), dtype='float16')

    for i in range(0,patch_number):
        data[i] = generateRandomPatch(d, patch_size_x, patch_size_y, patch_size_z)

    return data

# divide the full image into patchs
# todo : missing data if shape%patch_size is not 0
def generateFullPatchs(d, patch_size_x, patch_size_y, patch_size_z):
    patch_nb = int((d.shape[0]/patch_size_x)*(d.shape[1]/patch_size_y)*(d.shape[2]/patch_size_z))
    data = np.empty((patch_nb, patch_size_x, patch_size_y, patch_size_z), dtype='float16')
    i = 0
    for x in range(0,d.shape[0], patch_size_x):
        for y in range(0, d.shape[1], patch_size_y):
            for z in range(0,d.shape[2], patch_size_z):
                data[i] = extractPatch(d, patch_size_x, patch_size_y, patch_size_z, x, y, z)
                i = i+1

    return data

def generateFullPatchsPlus(d, patch_size_x, patch_size_y, patch_size_z, dx, dy, dz):
    patch_nb = int((d.shape[0]/dx)*(d.shape[1]/dy)*(d.shape[2]/dz))
    data = np.empty((patch_nb, patch_size_x, patch_size_y, patch_size_z), dtype='float16')
    i = 0
    for x in range(0,d.shape[0]-dx, dx):
        for y in range(0, d.shape[1]-dy, dy):
            for z in range(0,d.shape[2]-dz, dz):
                data[i] = extractPatch(d, patch_size_x, patch_size_y, patch_size_z, x, y, z)
                i = i+1

    return data

def noNeg(x):
    if(x>0):
        return x
    else:
        return 0

def generateFullPatchsCentered(d, patch_size_x, patch_size_y, patch_size_z):
    patch_nb = int(2*(d.shape[0]/patch_size_x)*2*(d.shape[1]/patch_size_y)*2*(d.shape[2]/patch_size_z))
    data = np.empty((patch_nb, patch_size_x, patch_size_y, patch_size_z), dtype='float16')
    i = 0
    psx = int(patch_size_x/2)
    psy = int(patch_size_y/2)
    psz = int(patch_size_z/2)
    for x in range(-16,d.shape[0]-16, psx):
        for y in range(-16, d.shape[1]-16, psy):
            for z in range(-16,d.shape[2]-16, psz):
                patch = np.zeros((psx,psy,psz), dtype='float16')
                patch = d[noNeg(x):noNeg(x)+patch_size_x,noNeg(y):noNeg(y)+patch_size_y,noNeg(z):noNeg(z)+patch_size_z]
                data[i] = patch
                i = i+1
    return data

# ----- Patch Extraction Generator -----
# Generator of random patchs of size 32*32*32
def generatorRandomPatchs32(features, labels, batch_size):
    batch_features = np.zeros((batch_size, 32, 32, 32, features.shape[4]), dtype='float16')
    batch_labels = np.zeros((batch_size, 32, 32, 32, labels.shape[4]), dtype='float16')

    while True:
        for i in range(batch_size):
            id = randint(0,features.shape[0]-1)
            x = randint(0, features.shape[1]-32)
            y = randint(0, features.shape[2]-32)
            z = randint(0, features.shape[3]-32)

            batch_features[i]   = extractPatch(features[id], 32, 32, 32, x, y, z)
            batch_labels[i]     = extractPatch(labels[id], 32, 32, 32, x, y, z)

        yield batch_features, batch_labels

# Generator of random patchs of size 32*32*32 and 16*16*16
def generatorRandomPatchs3216(features, labels, batch_size):
    batch_features = np.zeros((batch_size, 32, 32, 32, features.shape[4]), dtype='float16')
    batch_labels = np.zeros((batch_size, 16, 16, 16, labels.shape[4]), dtype='float16')

    while True:
        for i in range(batch_size):
            id = randint(0,features.shape[0]-1)
            x = randint(0, features.shape[1]-32)
            y = randint(0, features.shape[2]-32)
            z = randint(0, features.shape[3]-32)

            batch_features[i]   = extractPatch(features[id], 32, 32, 32, x, y, z)
            batch_labels[i]     = extractPatch(labels[id], 16, 16, 16, x+16, y+16, z+16)

        yield batch_features, batch_labels

# ----- Image Reconstruction -----
# Recreate the image from patchs
def fullPatchsToImage(image,patchs):
    i = 0
    for x in range(0,image.shape[0], patchs.shape[1]):
        for y in range(0, image.shape[1], patchs.shape[2]):
            for z in range(0,image.shape[2], patchs.shape[3]):
                image[x:x+patchs.shape[1],y:y+patchs.shape[2],z:z+patchs.shape[3]] = patchs[i,:,:,:,0]
                i = i+1
    return image

def fullPatchsPlusToImage(image,patchs, dx, dy, dz):
    div = np.zeros(image.shape)
    one = np.ones((patchs.shape[1],patchs.shape[2],patchs.shape[3]))

    i = 0
    for x in range(0,image.shape[0]-dx, dx):
        for y in range(0, image.shape[1]-dy, dy):
            for z in range(0,image.shape[2]-dz, dz):
                div[x:x+patchs.shape[1],y:y+patchs.shape[2],z:z+patchs.shape[3]] += one
                image[x:x+patchs.shape[1],y:y+patchs.shape[2],z:z+patchs.shape[3]] = patchs[i,:,:,:,0]
                i = i+1

    image = image/div

    return image