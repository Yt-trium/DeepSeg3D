# ------------------------------------------------------------ #
#
# file : training.py
# author : CM
# Launch the training
#
# ------------------------------------------------------------ #
from readConfig import readConfig
from dataAccessor import readDataset, reshapeDataset

config = readConfig("config.txt")

dataset = readDataset(config["dataset_train_gd_path"],
                      config["dataset_train_size"],
                      config["image_size_x"],
                      config["image_size_y"],
                      config["image_size_z"])

print(reshapeDataset(dataset).shape)