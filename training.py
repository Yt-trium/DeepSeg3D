# ------------------------------------------------------------ #
#
# file : training.py
# author : CM
# Launch the training
#
# ------------------------------------------------------------ #
import os
import sys
from time import sleep
from readConfig import readConfig
from dataAccessor import readDataset, reshapeDataset, generateRandomPatchs, generateFullPatchs, generatorRandomPatchs32
from models.unet import unet_1
from models.metrics import sensitivity, specificity
from keras.optimizers import Adam
from keras.utils.training_utils import multi_gpu_model
from keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint

from keras import backend as K
K.set_image_dim_ordering("tf")

config_filename = sys.argv[1]
if(not os.path.isfile(config_filename)):
    sys.exit(1)

config = readConfig(config_filename)

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
# model = multi_gpu_model(model,2)
model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=[sensitivity, specificity])

# model.summary()

print("Start training")

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
csv_logger = CSVLogger('./logs/training.log')
checkpoint = ModelCheckpoint(filepath='./logs/model-{epoch:03d}.h5')

model.fit_generator(generatorRandomPatchs32(train_mra_dataset, train_gd_dataset, config["batch_size"]),
                    steps_per_epoch=config["steps_per_epoch"], epochs=config["epochs"],
                    verbose=1, callbacks=[tensorboard, csv_logger, checkpoint])

model.save('./logs/model-final.h5')

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