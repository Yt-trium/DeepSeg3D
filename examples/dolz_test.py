# ------------------------------------------------------------ #
#
# file : examples/dolz_test.py
# author : CM
# Example of testing on Bullitt dataset using Jose Dolz FCNN
# Original article : 3D fully convolutional networks
# for subcortical segmentation in MRI: A large-scale study
#
# ------------------------------------------------------------ #


# ----- Configuration and model creation -----
# Set ordering : tensorflow (channel last)
import os
import sys

from utils.config.read import readConfig
from utils.io.read import readRawDataset, reshapeDataset
from utils.io.write import npToNii
from utils.learning.patch.extraction import generateFullPatchsCentered
from utils.learning.patch.reconstruction import dolzReconstruction
from utils.preprocessing.normalisation import intensityNormalisation

from keras import backend as K,  models

# ----- Configuration and model load -----
# Set ordering : tensorflow (channel last)
K.set_image_dim_ordering("tf")

# Check config file exist
config_filename = sys.argv[1]
if(not os.path.isfile(config_filename)):
    sys.exit(1)
# Load config
config = readConfig(config_filename)

filename = sys.argv[2]
if(not os.path.isfile(filename)):
    sys.exit(1)

# Load model
print("Load model")
model = models.load_model(filename)

# ----- Dataset load -----
print("Loading testing dataset")

test_gd_dataset = readRawDataset(config["dataset_test_gd_path"],
                                  config["dataset_test_size"],
                                  config["image_size_x"],
                                  config["image_size_y"],
                                  config["image_size_z"],
                                  'uint16')

print("Testing ground truth dataset shape", test_gd_dataset.shape)
print("Testing ground truth dataset dtype", test_gd_dataset.dtype)

test_in_dataset = readRawDataset(config["dataset_test_mra_path"],
                                  config["dataset_test_size"],
                                  config["image_size_x"],
                                  config["image_size_y"],
                                  config["image_size_z"],
                                  'uint16')

print("Training input image dataset shape", test_in_dataset.shape)
print("Training input image dataset dtype", test_in_dataset.dtype)

# ----- PreProcessing -----
# Intensity normalisation
print("Apply intensity normalisation to input image dataset")

test_in_dataset = intensityNormalisation(test_in_dataset, 'float32')

print("Testing input image dataset dtype", test_in_dataset.dtype)

# ----- Evaluation and prediction -----
print("Generate prediction")

from keras.utils import to_categorical
import numpy as np
for count in range(0,test_in_dataset.shape[0]):
    patchs_in = generateFullPatchsCentered(test_in_dataset[count], 32, 32, 32)
    patchs_in = reshapeDataset(patchs_in)

    prediction = model.predict(patchs_in)

    segmentation = dolzReconstruction(test_in_dataset[count], prediction)

    print(str(count + 1) + '/' + str(config["dataset_test_size"]))

    npToNii(segmentation, (str(count + 1).zfill(2) + ".nii.gz"))

    """
    patchs_gd = generateFullPatchsCentered(test_gd_dataset[count], 32, 32, 32)
    patchs_gd = reshapeDataset(patchs_gd)
    patchs_gd = generateFullPatchs(test_gd_dataset[count], 32, 32, 32)
    patchs_gd = reshapeDataset(patchs_gd)
    patchs_mra = generateFullPatchs(test_mra_dataset[count], 32, 32, 32)
    patchs_mra = reshapeDataset(patchs_mra)

    print(str(count+1)+'/'+str(config["dataset_test_size"]))

    print(model.evaluate(patchs_mra,patchs_gd))

    prediction = model.predict(patchs_mra)
    prediction.reshape(prediction.shape[0],prediction.shape[1],prediction.shape[2],prediction.shape[3])

    image = np.empty((config["image_size_x"], config["image_size_y"], config["image_size_z"]))

    fullPatchsToImage(image,prediction)

    npToNii(image,(str(count+1).zfill(2)+".nii.gz"))
    """