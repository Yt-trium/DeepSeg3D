# ------------------------------------------------------------ #
#
# file : training.py
# author : CM
# Launch the training
#
# ------------------------------------------------------------ #
from time import sleep
from readConfig import readConfig
from dataAccessor import readDataset, reshapeDataset, generateRandomPatchs, generateFullPatchs, generatorRandomPatchs32
from models.unet import unet_1

from keras import backend as K
K.set_image_dim_ordering("tf")

config = readConfig("config.txt")

print("Loading training dataset")

train_gd_dataset = readDataset(config["dataset_train_gd_path"],
                               config["dataset_train_size"],
                               config["image_size_x"],
                               config["image_size_y"],
                               config["image_size_z"])
train_gd_dataset = reshapeDataset(train_gd_dataset)

train_mra_dataset = readDataset(config["dataset_train_mra_path"],
                                config["dataset_train_size"],
                                config["image_size_x"],
                                config["image_size_y"],
                                config["image_size_z"])
train_mra_dataset = reshapeDataset(train_mra_dataset)

print("Generate model")

model = unet_1(config["patch_size_x"],config["patch_size_y"],config["patch_size_z"])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# model.summary()

print("Start training")

model.fit_generator(generatorRandomPatchs32(train_gd_dataset, train_mra_dataset, 8), steps_per_epoch=2, epochs=1, verbose=1)

print("Saving results")

model.save_weights("model_weights.h5")

"""
# Validation
# free dataset memory
train_gd_dataset = None
train_mra_dataset = None

valid_gd_dataset = readDataset(config["dataset_valid_gd_path"],
                               config["dataset_valid_size"],
                               config["image_size_x"],
                               config["image_size_y"],
                               config["image_size_z"])
valid_gd_dataset = reshapeDataset(valid_gd_dataset)

valid_mra_dataset = readDataset(config["dataset_valid_mra_path"],
                                config["dataset_valid_size"],
                                config["image_size_x"],
                                config["image_size_y"],
                                config["image_size_z"])
valid_mra_dataset = reshapeDataset(valid_mra_dataset)
"""