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

from utils.config.read import readConfig
from utils.io.read import getDataset


class DeepSeg3D:
    # Flags to check if everything is done
    train_loaded = False
    valid_loaded = False
    test_loaded  = False

    # Dataset path
    train_gd_path = "#"
    train_in_path = "#"
    valid_gd_path = "#"
    valid_in_path = "#"
    test_gd_path  = "#"
    test_in_path  = "#"

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
    def load_train(self, size, type=None):
        print("[DeepSeg3D]", "load_train")
        self.train_gd = getDataset(self.train_gd_path, size, type)
        self.train_in = getDataset(self.train_in_path, size, type)
        self.train_loaded = True
        print(self.train_gd.shape, self.train_gd.dtype)

    def load_valid(self, size, type=None):
        print("[DeepSeg3D]", "load_valid")
        self.valid_gd = getDataset(self.valid_gd_path, size, type)
        self.valid_in = getDataset(self.valid_in_path, size, type)
        self.valid_loaded = True
        print(self.valid_gd.shape, self.train_gd.dtype)

    def load_test(self, size, type=None):
        print("[DeepSeg3D]", "load_test")
        self.test_gd = getDataset(self.test_gd_path, size, type)
        self.test_in = getDataset(self.test_in_path, size, type)
        self.test_loaded = True
        print(self.test_gd.shape, self.test_gd.dtype)


    def load_model(self):
        print("[DeepSeg3D]", "load_model")

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

    deepseg.train_gd_path = config["dataset_train_gd_path"]
    deepseg.train_in_path = config["dataset_train_mra_path"]
    deepseg.valid_gd_path = config["dataset_valid_gd_path"]
    deepseg.valid_in_path = config["dataset_valid_mra_path"]

    deepseg.load_train(config["dataset_train_size"], 'float16')
    deepseg.load_valid(config["dataset_valid_size"], 'float16')

    deepseg.logs_folder = config["logs_folder"]

    deepseg.train(config["epochs"])
