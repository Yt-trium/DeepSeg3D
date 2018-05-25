# ------------------------------------------------------------ #
#
# file : training.py
# author : CM
# Launch the training
#
# ------------------------------------------------------------ #
from readConfig import readConfig
from dataAccessor import readDataset, reshapeDataset, generateRandomPatchs, generateFullPatchs
from models.unet import unet_1

from keras import backend as K
K.set_image_dim_ordering("tf")

config = readConfig("config.txt")

dataset = readDataset(config["dataset_train_gd_path"],
                      config["dataset_train_size"],
                      config["image_size_x"],
                      config["image_size_y"],
                      config["image_size_z"])

batch = generateRandomPatchs(dataset[0], 32, 32, 32, 100)

print(batch.shape)

batch = generateFullPatchs(dataset[0], 32, 32, 32)

print(batch.shape)

model = unet_1(config["image_size_x"],config["image_size_y"],config["image_size_z"])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# model.summary()