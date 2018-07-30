# ------------------------------------------------------------ #
#
# file : tools/prediction_mixer.py
# author : CM
# Combine results to make a final better prediction
#
# ------------------------------------------------------------ #
import os
import sys
import nibabel as nib
import numpy as np

from utils.io.read import getAffine
from utils.io.write import npToNiiAffine

files = []

for i in range(1, len(sys.argv)):
    files.append(sys.argv[i])

    if not os.path.isfile(files[i-1]):
        sys.exit("invalid argument")

image = nib.load(files[0])
dtype = image.get_data_dtype()
affine = image.affine
image_size = (image.shape[0], image.shape[1], image.shape[2])

images = np.empty((len(files), image.shape[0], image.shape[1], image.shape[2])).astype(dtype)
del image

count = 0
for i in files:
    print(i)
    images[count, :, :, :] = nib.load(i).get_data().reshape(image_size)
    count += 1

addition = np.zeros(image_size).astype(dtype)
union = np.zeros(image_size).astype(dtype)

# operation : Union, Intersection, Addition, Average
for i in images:
    for x in range(image_size[0]):
        for y in range(image_size[1]):
            for z in range(image_size[2]):
                addition[x,y,z] = addition[x,y,z] + i[x,y,z]
                union[x,y,z] = union[x,y,z] or i[x,y,z]

npToNiiAffine(addition, affine, "addition.nii.gz")
addition = addition / len(files)
npToNiiAffine(addition, affine, "average.nii.gz")
npToNiiAffine(union, affine, "union.nii.gz")
