# ------------------------------------------------------------ #
#
# file : test.py
# author : CM
# Tests
#
# ------------------------------------------------------------ #
import numpy as np
from readConfig import readConfig
from dataAccessor import readDataset, reshapeDataset, generateFullPatchs, fullPatchsToImage, npToNii, niiToNp

config = readConfig("config.txt")

# ----- Image Patchs  -----
# Test to generate patchs and reconstruct the image from them
train_gd_dataset = readDataset(config["dataset_train_gd_path"],
                               1,
                               config["image_size_x"],
                               config["image_size_y"],
                               config["image_size_z"])

patchs = generateFullPatchs(train_gd_dataset[0], 32, 32, 32)
patchs = reshapeDataset(patchs)

image = np.empty((448, 448, 128))
fullPatchsToImage(image,patchs)

npToNii(image,"test.nii.gz")

image1 = train_gd_dataset[0]
image2 = niiToNp("test.nii.gz")

print(np.array_equal(image1,image2))

print(train_gd_dataset[0].max())
print(train_gd_dataset[0].min())
print(train_gd_dataset[0].mean())