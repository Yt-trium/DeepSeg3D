# ------------------------------------------------------------ #
#
# file : training.py
# author : CM
# Launch the training
#
# ------------------------------------------------------------ #
import os
import sys
import numpy as np
from time import sleep
from readConfig import readConfig
from dataAccessor import readDataset, reshapeDataset, generateRandomPatchs, generateFullPatchs, generatorRandomPatchs32
from models.unet import unet_1, unet_2, unet_3, cunet_1
from models.metrics import sensitivity, specificity
from models.losses import dice_coef, dice_coef_loss, jaccard_distance_loss
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.utils.training_utils import multi_gpu_model
from keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint, LearningRateScheduler

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

model = unet_2(config["patch_size_x"],config["patch_size_y"],config["patch_size_z"])
# plot_model(model, to_file='model.png')
# model = multi_gpu_model(model,2)
model.compile(optimizer=Adam(lr=1e-4), loss=jaccard_distance_loss, metrics=[dice_coef, sensitivity, specificity])

# model.summary()

print("Start training")

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
csv_logger = CSVLogger('./logs/training.log')
checkpoint = ModelCheckpoint(filepath='./logs/model-{epoch:03d}.h5')

def learning_rate_schedule(initial_lr=1e-4, decay_factor=0.99, step_size=1):
    def schedule(epoch):
        x = initial_lr * (decay_factor ** np.floor(epoch / step_size))
        print("Learning rate : ",x)
        return x
    return LearningRateScheduler(schedule)

# lr_sched = learning_rate_schedule(initial_lr=1e-3, decay_factor=0.95, step_size=1)
lr_sched = learning_rate_schedule(initial_lr=1e-4, decay_factor=0.99, step_size=1)


model.fit_generator(generatorRandomPatchs32(train_mra_dataset, train_gd_dataset, config["batch_size"]),
                    steps_per_epoch=config["steps_per_epoch"], epochs=config["epochs"],
                    verbose=1, callbacks=[tensorboard, csv_logger, checkpoint, lr_sched])

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