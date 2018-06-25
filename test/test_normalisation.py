# ------------------------------------------------------------ #
#
# file : test/test_normalisation.py
# author : CM
# Tests normalisation functions
#
# ------------------------------------------------------------ #
import os

import matplotlib.pyplot as plt
import nibabel as nib

from utils.preprocessing.normalisation import intensityNormalisationFeatureScaling, intensityMaxClipping, intensityProjection

files = os.listdir("../Datasets/Dataset/train_Images/")
files.sort()

data = nib.load(os.path.join("../Datasets/Dataset/train_Images/", files[0])).get_data()

print(data.max())
print(data.min())
print(data.mean())

plt.hist(data.flatten())
plt.show()

data = intensityMaxClipping(data, 250, data.dtype)

plt.hist(data.flatten())
plt.show()

data = intensityNormalisationFeatureScaling(data, data.dtype)

plt.hist(data.flatten())
plt.show()

data = intensityProjection(data, 3, data.dtype)

plt.hist(data.flatten())
plt.show()