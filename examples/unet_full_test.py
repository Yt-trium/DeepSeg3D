# ------------------------------------------------------------ #
#
# file : examples/unet_full_test.py
# author : CM
# Example of training on Bullitt dataset using 3D unet using
# full images and no patchs
#
# ------------------------------------------------------------ #

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
model = models.load_model(filename, custom_objects={'dice_loss':dice_loss,
                                                    'f1':f1,
                                                    'sensitivity':sensitivity,
                                                    'specificity':specificity,
                                                    'precision':precision})

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

print(model.evaluate(test_in_dataset, test_gd_dataset))
prediction = model.predict(test_in_dataset)

for count in prediction.shape[0]:
    print(str(count + 1) + '/' + str(config["dataset_test_size"]))
    npToNiiAffine(prediction[count], getAffine(config["dataset_test_gd_path"],), (str(count + 1).zfill(2) + ".nii.gz"))