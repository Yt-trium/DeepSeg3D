# ------------------------------------------------------------ #
#
# file : examples/unet_test.py
# author : CM
# Example of testing on Bullitt dataset using 3D unet
#
# ------------------------------------------------------------ #

# ----- Configuration and model creation -----
# Set ordering : tensorflow (channel last)
import os
import sys

from utils.config.read import readConfig
from utils.io.read import readRawDataset, reshapeDataset, getAffine
from utils.io.write import npToNii, npToNiiAffine
from utils.learning.losses import dice_coef_loss, dice_coef, dice_coef_loss_, dice_coef_, jaccard_distance_loss, dice_loss
from utils.learning.metrics import sensitivity, specificity, precision, f1
from utils.learning.patch.extraction import generateFullPatchsCentered, generateFullPatchs
from utils.learning.patch.reconstruction import dolzReconstruction, fullPatchsToImage
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
model = models.load_model(filename, custom_objects={'sensitivity':sensitivity,'specificity':specificity,'precision':precision,
                                                    'dice_coef_loss':dice_coef_loss,'dice_coef':dice_coef,
                                                    'dice_coef_loss_':dice_coef_loss_,'dice_coef_':dice_coef_,
                                                    'jaccard_distance_loss':jaccard_distance_loss, 'dice_loss':dice_loss,
                                                    'f1':f1})

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

for count in range(0,test_in_dataset.shape[0]):
    patchs_in = generateFullPatchsCentered(test_in_dataset[count], config["patch_size_x"],config["patch_size_y"],config["patch_size_z"])
    patchs_in = reshapeDataset(patchs_in)

    patchs_gd = generateFullPatchsCentered(test_gd_dataset[count], config["patch_size_x"],config["patch_size_y"],config["patch_size_z"])
    patchs_gd = reshapeDataset(patchs_gd)

    print(model.evaluate(patchs_in, patchs_gd))
    prediction = model.predict(patchs_in)

    label_selector = [slice(None)] + [slice(int(config["patch_size_x"]/4), int(3*(config["patch_size_x"]/4)))] + \
                     [slice(int(config["patch_size_y"] / 4), int(3 * (config["patch_size_y"] / 4)))] + \
                     [slice(int(config["patch_size_z"] / 4), int(3 * (config["patch_size_z"] / 4)))] + [slice(None)]
    prediction = prediction[label_selector]

    segmentation = fullPatchsToImage(test_in_dataset[count], prediction)

    print(str(count + 1) + '/' + str(config["dataset_test_size"]))
    npToNiiAffine(segmentation, getAffine(config["dataset_test_gd_path"],), (str(count + 1).zfill(2) + ".nii.gz"))