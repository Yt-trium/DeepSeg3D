# ------------------------------------------------------------ #
#
# file : testing.py
# author : CM
# Test the trained model
#
# ------------------------------------------------------------ #
import sys
import os.path
import numpy as np
from dataAccessor import readDataset, reshapeDataset, generateFullPatchs, fullPatchsToImage, npToNii
from readConfig import readConfig
from models.unet import unet_1
from models.metrics import sensitivity, specificity
from models.losses import dice_coef, dice_coef_loss
from keras.optimizers import Adam
from keras import backend as K, models

K.set_image_dim_ordering("tf")

config_filename = sys.argv[1]
if(not os.path.isfile(config_filename)):
    sys.exit(1)

config = readConfig(config_filename)

filename = sys.argv[2]
if(not os.path.isfile(filename)):
    sys.exit(1)

print("Loading test dataset")

test_mra_dataset = readDataset(config["dataset_test_mra_path"],
                                config["dataset_test_size"],
                                config["image_size_x"],
                                config["image_size_y"],
                                config["image_size_z"])

print("Loading model and trained weights")

model = models.load_model(filename, custom_objects={'sensitivity':sensitivity,'specificity':specificity,
                                                    'dice_coef_loss':dice_coef_loss,'dice_coef':dice_coef})

print("Generate prediction")

count = 0
for mra in test_mra_dataset:
    count = count + 1
    print(str(count)+'/'+str(config["dataset_test_size"]))
    patchs = generateFullPatchs(mra, 32, 32, 32)
    patchs = reshapeDataset(patchs)

    prediction = model.predict(patchs)
    prediction.reshape(prediction.shape[0],prediction.shape[1],prediction.shape[2],prediction.shape[3])

    image = np.empty((448, 448, 128))

    fullPatchsToImage(image,prediction)

    npToNii(image,(str(count).zfill(2)+".nii.gz"))