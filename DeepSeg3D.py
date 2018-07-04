# ------------------------------------------------------------ #
#
# file : DeepSeg3D.py
# author : CM
# The main program, the only one to be called by the final user
#
# ------------------------------------------------------------ #
import os
import sys
import tensorflow as tf
import tensorboard as tb

from models.unet import *
from utils.config.read import readConfig
from utils.io.read import readDatasetPart

class DeepSeg3D:
    # Flags to check if everything is done
    train_loaded = False
    valid_loaded = False
    test_loaded  = False

    # Dataset path
    in_path = "#"
    gd_path = "#"
    dataset_size = (0, 0, 0)

    # Dataset
    train_gd = None
    train_in = None
    valid_gd = None
    valid_in = None
    test_gd  = None
    test_in  = None

    # Logs
    logs_folder = "#"

    # Constructor
    def __init__(self):
        print("[DeepSeg3D]", "__init__")
        self.sess = tf.Session()

    # Dataset load
    def load_train(self, type=None):
        print("[DeepSeg3D]", "load_train", type)
        self.train_gd = readDatasetPart(self.gd_path, 0, self.dataset_size[0], type)
        self.train_in = readDatasetPart(self.in_path, 0, self.dataset_size[0], type)
        self.train_loaded = True
        print("[DeepSeg3D]", "dataset shape", self.train_gd.shape, self.train_gd.dtype)

    def load_valid(self, type=None):
        print("[DeepSeg3D]", "load_valid", type)
        self.valid_gd = readDatasetPart(self.gd_path, self.dataset_size[0], self.dataset_size[1], type)
        self.valid_in = readDatasetPart(self.in_path, self.dataset_size[0], self.dataset_size[1], type)
        self.valid_loaded = True
        print("[DeepSeg3D]", "dataset shape", self.valid_gd.shape, self.train_gd.dtype)

    def load_test(self, type=None):
        print("[DeepSeg3D]", "load_test", type)
        self.test_gd = readDatasetPart(self.gd_path, self.dataset_size[0]+self.dataset_size[1], self.dataset_size[2], type)
        self.test_in = readDatasetPart(self.in_path, self.dataset_size[0]+self.dataset_size[1], self.dataset_size[2], type)
        self.test_loaded = True
        print("[DeepSeg3D]", "dataset shape", self.test_gd.shape, self.test_gd.dtype)


    # Model load
    def load_model(self, name, p):
        print("[DeepSeg3D]", "load_model", name, p)
        return globals()[name](p[0], p[1], p[2])

    # Train the current loaded model
    def train(self, epochs):
        print("[DeepSeg3D]", "train")

        if not self.train_loaded or not self.valid_loaded:
            sys.exit("FATAL ERROR: train or valid dataset not loaded for training")

        try:
            for epoch in range(epochs):
                print("Epoch :", epoch+1, '/', epochs)



        except  KeyboardInterrupt:
            print("KeyboardInterrupt : Closing")

        self.sess.close()


# ------------------------------------------------------------ #
#
# Example of an instantiation
#
# ------------------------------------------------------------ #

# ----- DeepSeg3D instantiation -----
if __name__ == '__main__':
    # Check if config filename exist
    config_filename = sys.argv[1]
    if (not os.path.isfile(config_filename)):
        sys.exit("FATAL ERROR: configuration file doesn't exists")
    # Read config
    config = readConfig(config_filename)

    # ----- DeepSeg3D training -----
    deepseg = DeepSeg3D()

    deepseg.in_path = config["dataset_in_path"]
    deepseg.gd_path = config["dataset_gd_path"]

    deepseg.dataset_size = (config["dataset_train"], config["dataset_valid"], config["dataset_test"])

    deepseg.load_train('float16')
    deepseg.load_valid('float16')

    deepseg.load_model("unet_3_light", (config["train_patch_size_x"], config["train_patch_size_y"], config["train_patch_size_z"]))

    deepseg.logs_folder = config["logs_path"]

    deepseg.train(config["train_epochs"])
