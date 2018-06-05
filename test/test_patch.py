# ------------------------------------------------------------ #
#
# file : test/test_patch.py
# author : CM
# Tests
#
# ------------------------------------------------------------ #
import numpy as np
from utils.config.read import readConfig
from utils.io.read import readDataset, reshapeDataset, niiToNp
from utils.io.write import npToNii
from utils.learning.patch.extraction import generateFullPatchsPlus
from utils.learning.patch.reconstruction import fullPatchsPlusToImage

config = readConfig("config.txt")

# ----- Image Patchs  -----
# Test to generate patchs and reconstruct the image from them
train_gd_dataset = readDataset(config["dataset_train_gd_path"],
                               1,
                               config["image_size_x"],
                               config["image_size_y"],
                               config["image_size_z"])

patchs = generateFullPatchsPlus(train_gd_dataset[0], 32, 32, 32, 16, 16, 16)
patchs = reshapeDataset(patchs)

image = np.empty((config["image_size_x"], config["image_size_y"], config["image_size_z"]))
fullPatchsPlusToImage(image,patchs, 16, 16, 16)

npToNii(image,"test.nii.gz")

image1 = train_gd_dataset[0]
image2 = niiToNp("test.nii.gz")

print(np.array_equal(image1,image2))

print(train_gd_dataset[0].max())
print(train_gd_dataset[0].min())
print(train_gd_dataset[0].mean())