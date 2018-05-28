# ------------------------------------------------------------ #
#
# file : test.py
# author : CM
# Tests
#
# ------------------------------------------------------------ #
import numpy as np
from readConfig import readConfig
from dataAccessor import readDataset, reshapeDataset, generateFullPatchs, fullPatchsToImage, npToNii

config = readConfig("config.txt")

# ----- Image Patchs  -----
# Test to generate patchs and reconstruct the image from them
train_gd_dataset = readDataset(config["dataset_train_mra_path"],
                               1,
                               config["image_size_x"],
                               config["image_size_y"],
                               config["image_size_z"])

patchs = generateFullPatchs(train_gd_dataset[0], 32, 32, 32)
patchs = reshapeDataset(patchs)

image = np.empty((448, 448, 128))
fullPatchsToImage(image,patchs)

npToNii(image,"test.nii.gz")