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

avg_ = np.zeros(image_size).astype(dtype)
max_ = np.zeros(image_size).astype(dtype)
min_ = np.ones(image_size).astype(dtype)
# operations :
# Average (sum / number of files)
# Max (max of every files)
# Min (min of every files)

for i in images:
    for x in range(image_size[0]):
        for y in range(image_size[1]):
            for z in range(image_size[2]):
                avg_[x,y,z] = avg_[x,y,z] + i[x,y,z]
                max_[x,y,z] = max(max_[x,y,z], i[x,y,z])
                min_[x,y,z] = min(min_[x,y,z], i[x,y,z])
avg_ = avg_ / len(files)

npToNiiAffine(avg_, affine, "avg.nii.gz")
npToNiiAffine(max_, affine, "max.nii.gz")
npToNiiAffine(min_, affine, "min.nii.gz")
