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
from utils.io.read import readDataset, reshapeDataset
from utils.io.write import npToNii
from utils.learning.patch.extraction import generateFullPatchs
from utils.learning.patch.reconstruction import fullPatchsToImage
from utils.config.read import readConfig
from utils.learning.metrics import sensitivity, specificity, precision
from utils.learning.losses import dice_coef, dice_coef_loss, jaccard_distance_loss, dice_coef_, dice_coef_loss_
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

test_gd_dataset = readDataset(config["dataset_test_gd_path"],
                                config["dataset_test_size"],
                                config["image_size_x"],
                                config["image_size_y"],
                                config["image_size_z"])

test_mra_dataset = readDataset(config["dataset_test_mra_path"],
                                config["dataset_test_size"],
                                config["image_size_x"],
                                config["image_size_y"],
                                config["image_size_z"])

print("Loading model and trained weights")

model = models.load_model(filename, custom_objects={'sensitivity':sensitivity,'specificity':specificity,'precision':precision,
                                                    'dice_coef_loss':dice_coef_loss,'dice_coef':dice_coef,
                                                    'dice_coef_loss_':dice_coef_loss_,'dice_coef_':dice_coef_,
                                                    'jaccard_distance_loss': jaccard_distance_loss})

print("Generate prediction")

for count in range(0,test_mra_dataset.shape[0]):
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