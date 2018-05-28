# ------------------------------------------------------------ #
#
# file : testing.py
# author : CM
# Test the trained model
#
# ------------------------------------------------------------ #
import numpy as np
from dataAccessor import readDataset, reshapeDataset, generateFullPatchs, fullPatchsToImage, npToNii
from readConfig import readConfig
from models.unet import unet_1
from keras.optimizers import Adam
from keras import backend as K
K.set_image_dim_ordering("tf")

config = readConfig("config.txt")

print("Loading test dataset")

test_mra_dataset = readDataset(config["dataset_test_mra_path"],
                                config["dataset_test_size"],
                                config["image_size_x"],
                                config["image_size_y"],
                                config["image_size_z"])

print("Loading model and trained weights")

model = unet_1(config["patch_size_x"],config["patch_size_y"],config["patch_size_z"])
model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy')

model.load_weights("model_weights.h5")

print("Generate prediction")

count = 0
for mra in test_mra_dataset:
    patchs = generateFullPatchs(mra, 32, 32, 32)
    patchs = reshapeDataset(patchs)

    prediction = model.predict(patchs)
    prediction.reshape(prediction.shape[0],prediction.shape[1],prediction.shape[2],prediction.shape[3])

    image = np.empty((448, 448, 128))

    fullPatchsToImage(image,prediction)

    npToNii(image,(str(count).zfill(2)+".nii.gz"))
    count = count + 1