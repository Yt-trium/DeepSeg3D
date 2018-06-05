# ------------------------------------------------------------ #
#
# file : utils/learning/patch/reconstruction.py
# author : CM
# Function to reconstruct image from patch
#
# ------------------------------------------------------------

import numpy as np

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