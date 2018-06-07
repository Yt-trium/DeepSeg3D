# ------------------------------------------------------------ #
#
# file : examples/unet_train.py
# author : CM
# Example of training on Bullitt dataset using 3D unet
#
# ------------------------------------------------------------ #
import os
import sys

from models.unet import unet_3
from utils.config.read import readConfig
from utils.io.read import readRawDataset, reshapeDataset
from utils.learning.losses import dice_coef, dice_coef_, dice_coef_loss_, dice_coef_loss
from utils.learning.metrics import sensitivity, specificity, precision
from utils.learning.patch.extraction import generatorRandomPatchs
from utils.preprocessing.normalisation import intensityNormalisation
from utils.learning.callbacks import learningRateSchedule

from keras.optimizers import Adam
from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint
from keras import backend as K

# ----- Configuration and model creation -----
# Set ordering : tensorflow (channel last)
K.set_image_dim_ordering("tf")

# Check config file exist
config_filename = sys.argv[1]
if(not os.path.isfile(config_filename)):
    sys.exit(1)
# Load config
config = readConfig(config_filename)

# Generate model
print("Generate model")
model = unet_3(config["patch_size_x"],config["patch_size_y"],config["patch_size_z"])
model.compile(loss=dice_coef_loss, optimizer=Adam(lr=1e-4), metrics=[dice_coef, sensitivity, specificity, precision,
                                                                     dice_coef_, dice_coef_loss_, dice_coef_loss])

# Print model informations
model.summary()
print("Input shape", model.input_shape)
print("Output shape",model.output_shape)

# ----- Dataset load -----
print("Loading training dataset")

train_gd_dataset = readRawDataset(config["dataset_train_gd_path"],
                                  config["dataset_train_size"],
                                  config["image_size_x"],
                                  config["image_size_y"],
                                  config["image_size_z"],
                                  'uint16')

print("Training ground truth dataset shape", train_gd_dataset.shape)
print("Training ground truth dataset dtype", train_gd_dataset.dtype)

train_in_dataset = readRawDataset(config["dataset_train_mra_path"],
                                  config["dataset_train_size"],
                                  config["image_size_x"],
                                  config["image_size_y"],
                                  config["image_size_z"],
                                  'uint16')

print("Training input image dataset shape", train_in_dataset.shape)
print("Training input image dataset dtype", train_in_dataset.dtype)

print("Loading validation dataset")

valid_gd_dataset = readRawDataset(config["dataset_valid_gd_path"],
                                  config["dataset_valid_size"],
                                  config["image_size_x"],
                                  config["image_size_y"],
                                  config["image_size_z"],
                                  'uint16')

print("Validation ground truth dataset shape", valid_gd_dataset.shape)
print("Validation ground truth dataset dtype", valid_gd_dataset.dtype)

valid_in_dataset = readRawDataset(config["dataset_valid_mra_path"],
                                  config["dataset_valid_size"],
                                  config["image_size_x"],
                                  config["image_size_y"],
                                  config["image_size_z"],
                                  'uint16')

print("Validation input image dataset shape", valid_in_dataset.shape)
print("Validation input image dataset dtype", valid_in_dataset.dtype)

# ----- PreProcessing -----
# Intensity normalisation
print("Apply intensity normalisation to input image dataset")

train_in_dataset = intensityNormalisation(train_in_dataset, 'float32')
valid_in_dataset = intensityNormalisation(valid_in_dataset, 'float32')

print("Training input image dataset dtype", train_in_dataset.dtype)
print("Validation input image dataset dtype", valid_in_dataset.dtype)

train_in_dataset = reshapeDataset(train_in_dataset)
train_gd_dataset = reshapeDataset(train_gd_dataset)
valid_in_dataset = reshapeDataset(valid_in_dataset)
valid_gd_dataset = reshapeDataset(valid_gd_dataset)

# ----- Model training -----
# Callbacks
tensorboardCB  = TensorBoard(log_dir=config["logs_folder"], histogram_freq=0, write_graph=True, write_grads=True, write_images=True)
csvLoggerCB    = CSVLogger(str(config["logs_folder"]+'training.log'))
checkpointCB   = ModelCheckpoint(filepath=str(config["logs_folder"]+'model-{epoch:03d}.h5'))
learningRateCB = learningRateSchedule()

print("Training")
model.fit_generator(generator=generatorRandomPatchs(train_in_dataset, train_gd_dataset, config["batch_size"],
                                          config["patch_size_x"],config["patch_size_y"],config["patch_size_z"]),
                    steps_per_epoch=config["steps_per_epoch"], epochs=config["epochs"],
                    verbose=1, callbacks=[tensorboardCB, csvLoggerCB, checkpointCB, learningRateCB],
                    validation_data=generatorRandomPatchs(valid_in_dataset, valid_gd_dataset, config["batch_size"],
                                          config["patch_size_x"],config["patch_size_y"],config["patch_size_z"]),
                    validation_steps=config["steps_per_epoch"])