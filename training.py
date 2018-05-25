# ------------------------------------------------------------ #
#
# file : training.py
# author : CM
# Launch the training
#
# ------------------------------------------------------------ #
from readConfig import readConfig
from dataAccessor import readDataset, reshapeDataset, generateRandomPatchs, generateFullPatchs, generatorRandomPatchs32
from models.unet import unet_1

from keras import backend as K
K.set_image_dim_ordering("tf")

config = readConfig("config.txt")

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

model = unet_1(config["patch_size_x"],config["patch_size_y"],config["patch_size_z"])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# model.summary()

model.fit_generator(generatorRandomPatchs32(train_gd_dataset, train_mra_dataset, 8), steps_per_epoch=2, epochs=1, verbose=1)