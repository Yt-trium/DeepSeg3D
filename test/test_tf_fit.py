# ------------------------------------------------------------ #
#
# file : test/test_tf_fit.py
# author : CM
# Tests to replace model.fit and model.fit_generator with
# TensorFlow code
#
# ------------------------------------------------------------ #

import os
import sys

from models.unet import unet_3
from utils.config.read import readConfig
from utils.io.read import readRawDataset, reshapeDataset, readTrainValid
from utils.learning.losses import dice_coef, dice_coef_, dice_coef_loss_, dice_coef_loss, dice_loss
from utils.learning.metrics import sensitivity, specificity, precision, f1
from utils.learning.patch.extraction import generatorRandomPatchs
from utils.preprocessing.normalisation import intensityNormalisation
from utils.learning.callbacks import learningRateSchedule

from keras.optimizers import Adam
from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint
from keras import backend as K
import tensorflow as tf

# ----- Configuration and model creation -----
# Set ordering : tensorflow (channel last)
K.set_image_dim_ordering("tf")

# Check config file exist
config_filename = sys.argv[1]
if(not os.path.isfile(config_filename)):
    sys.exit(1)
# Load config
config = readConfig(config_filename)

# ----- Dataset load -----
train_gd_dataset, train_in_dataset, valid_gd_dataset, valid_in_dataset = readTrainValid(config)
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

class ModelUnet:
    def __init__(self, input_shape, output_shape):

        self._X = tf.placeholder(name="X", dtype=tf.float32, shape=input_shape)
        self._Y = tf.placeholder(name="Y", dtype=tf.float32, shape=output_shape)
        self._Y_pred = tf.one_hot(self._Y, 2)

        self.lr = tf.get_variable("learningRate", initializer=1e-4, trainable=False)

        self._hat_Y_pred = unet_3(32, 32, 32)(self._X)
        self._loss = - tf.reduce_mean(self._Y_pred*tf.log(self._hat_Y_pred+1e-10))

        self._hat_Y = tf.cast(tf.argmax(self._hat_Y_pred, axis=1), tf.float32)
        self._accuracy = tf.reduce_mean(tf.cast(tf.equal(self._hat_Y, self._Y), tf.float32))

        self._opt = tf.train.AdamOptimizer(self.lr).minimize(self._loss)

        self._saver = tf.train.Saver()
        self.sess = tf.Session()


    def close(self):
        self.sess.close()
        tf.reset_default_graph()

    def fit(self, X_train, Y_train):
        self.loss, self.accuracy, _ = self.sess.run([self._loss, self._accuracy, self._opt],
                                          feed_dict={self._X: X_train, self._Y: Y_train})

        if self.verbose:
            print("loss: %.3f \t accuracy: %.2f" % (self.loss, self.accuracy))

        return self.loss, self.accuracy

    def validate(self, X_valid, Y_valid):

        self.loss, self.accuracy = self.sess.run([self._loss,self. _accuracy], feed_dict={self._X:X_valid,self._Y:Y_valid})

        if self.verbose:
            print("VALIDATION=>")
            print("loss: %.3f \t accuracy: %.2f" % (self.loss, self.accuracy))
            print("=" * 100 + "\n")

        return self.loss, self.accuracy

    def predict(self, X_test):
        return self.sess.run(self._hat_Y, feed_dict={self._X: X_test})



# Generate model
print("Generate model")
model = unet_3(config["patch_size_x"],config["patch_size_y"],config["patch_size_z"])
#model.compile(loss=dice_loss, optimizer=Adam(lr=1e-4), metrics=[f1, dice_loss, dice_coef, sensitivity, specificity, precision,
#                                                                     dice_coef_, dice_coef_loss_, dice_coef_loss])
# Print model informations
model.summary()
print("Input shape", model.input_shape)
print("Output shape",model.output_shape)