# ------------------------------------------------------------ #
#
# file : DeepSeg3D.py
# author : CM
# The main program, the only one to be called by the final user
#
# ------------------------------------------------------------ #
import os
import sys
from time import time
import tensorflow as tf
import tensorboard as tb
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint
from keras import backend as K

from utils.preprocessing.normalisation import intensityNormalisation

K.set_image_dim_ordering("tf")

from models.unet import Unet_1, unet_3_light
from utils.config.read import readConfig
from utils.io.read import readDatasetPart, reshapeDataset
from utils.learning.callbacks import learningRateSchedule
from utils.learning.losses import dice_loss
from utils.learning.metrics import f1, sensitivity, specificity, precision
from utils.learning.patch.extraction import randomPatchsAugmented


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

    # Train
    model = None
    patchs_size = (0, 0, 0)

    # Constructor
    def __init__(self):
        print("[DeepSeg3D]", "__init__")
        self.id = str(time())

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
    #def load_model(self, name, p):
    #    print("[DeepSeg3D]", "load_model", name, p)
    #    self.model = globals()[name](p[0], p[1], p[2])

    # Train with tensorflow
    def train_tf(self, epochs, steps_per_epoch, batch_size):
        self.sess = tf.Session()
        print("[DeepSeg3D]", "train")

        # Check dataset loaded
        if not self.train_loaded or not self.valid_loaded:
            sys.exit("FATAL ERROR: train or valid dataset not loaded for training")

        if self.patchs_size == self.train_in.shape[1:]:
            print("[DeepSeg3D]", "training on full images", self.patchs_size)
            train_full_images = True
        else:
            print("[DeepSeg3D]", "training on", self.patchs_size, "patchs")
            train_full_images = False

        # Tensorflow
        tf_in_ph = tf.placeholder(name="in_ph", dtype=tf.float32, shape=[None, self.patchs_size[0], self.patchs_size[1], self.patchs_size[2], 1])
        tf_gd_ph = tf.placeholder(name="gd_ph", dtype=tf.float32, shape=[None, self.patchs_size[0], self.patchs_size[1], self.patchs_size[2], 1])
        tf_lr = tf.get_variable("learningRate", initializer=1e-4, trainable=False)

        self.model = Unet_1()(tf_in_ph)

        y_true_f = tf.reshape(tf_gd_ph, [-1])
        y_pred_f = tf.reshape(self.model, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        tf_dice_loss = - (2. * intersection + 1e-6) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1e-6)

        tf_optimizer = tf.train.AdamOptimizer(tf_lr).minimize(tf_dice_loss)

        # saver = tf.train.Saver()
        # Summary
        tf.summary.scalar('dice loss', tf_dice_loss)
        tf.summary.scalar('learning rate', tf_lr)
        summary_merged = tf.summary.merge_all()
        train_summary_writer = tf.summary.FileWriter(self.logs_folder + self.id + "/train", self.sess.graph)
        valid_summary_writer = tf.summary.FileWriter(self.logs_folder + self.id + "/valid")

        self.sess.run(tf.global_variables_initializer())

        try:
            for epoch in range(epochs):
                print("Epoch :", epoch+1, '/', epochs)

                print("learning_rate :", self.sess.run(tf_lr))

                if train_full_images:
                    for (x, y) in zip(self.train_in, self.train_gd):
                        x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2], 1)
                        y = y.reshape(1, y.shape[0], y.shape[1], y.shape[2], 1)

                        loss, _ = self.sess.run([tf_dice_loss, tf_optimizer], feed_dict={tf_in_ph: x, tf_gd_ph: y})
                        print("dice_loss {}".format(loss), end="\r")

                    # Valid
                    xv = self.valid_in
                    yv = self.valid_gd
                    xv = xv.reshape(xv.shape[0], xv.shape[1], xv.shape[2], xv.shape[3], 1)
                    yv = yv.reshape(yv.shape[0], yv.shape[1], yv.shape[2], yv.shape[3], 1)
                else:
                    for sub_epoch in range(steps_per_epoch):

                        x, y = randomPatchsAugmented(self.train_in, self.train_gd, batch_size, self.patchs_size, self.patchs_size)

                        loss, _ = self.sess.run([tf_dice_loss, tf_optimizer], feed_dict={tf_in_ph: x, tf_gd_ph: y})
                        print("dice_loss {}".format(loss), end="\r")

                    # Valid
                    xv, yv = randomPatchsAugmented(self.valid_in, self.valid_gd, batch_size, self.patchs_size,
                                                   self.patchs_size)
                print()

                summary, loss = self.sess.run([summary_merged, tf_dice_loss], feed_dict={tf_in_ph: x, tf_gd_ph: y})
                train_summary_writer.add_summary(summary, epoch)

                # validation
                summary, loss = self.sess.run([summary_merged, tf_dice_loss], feed_dict={tf_in_ph: xv, tf_gd_ph: yv})
                print("validation", "loss", loss)

                valid_summary_writer.add_summary(summary, epoch)

                self.sess.run(tf_lr.assign(tf_lr*0.99))

        except  KeyboardInterrupt:
            print("KeyboardInterrupt : Closing")

        self.sess.close()

    # Train with keras
    def train_k(self, epochs, steps_per_epoch, batch_size):
        logs_path = self.logs_folder + self.id
        tensorboardCB = TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_grads=True, write_images=True)
        csvLoggerCB = CSVLogger(logs_path + '/training.log')
        checkpointCB = ModelCheckpoint(filepath=logs_path + '/model-{epoch:03d}.h5')
        bestModelCB = ModelCheckpoint(filepath=logs_path + '/model-best.h5', verbose=1, save_best_only=True, mode='max')
        learningRateCB = learningRateSchedule(initialLr=1e-4, decayFactor=0.99)

        self.train_in = intensityNormalisation(self.train_in, 'float32')
        self.valid_in = intensityNormalisation(self.valid_in, 'float32')

        self.model = unet_3_light(self.patchs_size[0], self.patchs_size[1], self.patchs_size[2])
        self.model.compile(loss=dice_loss, optimizer=Adam(lr=1e-4),
                      metrics=[sensitivity, specificity, precision])

        self.train_in = reshapeDataset(self.train_in)
        self.train_gd = reshapeDataset(self.train_gd)
        self.valid_in = reshapeDataset(self.valid_in)
        self.valid_gd = reshapeDataset(self.valid_gd)

        self.model.fit(x=self.train_in, y=self.train_gd, verbose=2, batch_size=batch_size,
          callbacks=[tensorboardCB, csvLoggerCB, checkpointCB, bestModelCB, learningRateCB],
          epochs=epochs,
          validation_data=(self.valid_in, self.valid_gd))

# ------------------------------------------------------------ #
#
# Example of an instantiation
#
# ------------------------------------------------------------ #

# ----- DeepSeg3D instantiation -----
if __name__ == '__main__':
    # Only show one GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    # TRAIN
    if (len(sys.argv) == 2):
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
        deepseg.patchs_size = (config["train_patch_size_x"], config["train_patch_size_y"], config["train_patch_size_z"])

        deepseg.dataset_size = (config["dataset_train"], config["dataset_valid"], config["dataset_test"])

        deepseg.load_train('uint16')
        deepseg.load_valid('uint16')

        # deepseg.load_model("unet_3_light", deepseg.patchs_size)

        deepseg.logs_folder = config["logs_path"]

        deepseg.train_k(config["train_epochs"], config["train_steps_per_epoch"], config["train_batch_size"])
    # TEST
    elif (len(sys.argv) == 3):
        # Check if config filename exist
        config_filename = sys.argv[1]
        if (not os.path.isfile(config_filename)):
            sys.exit("FATAL ERROR: configuration file doesn't exists")
        # Read config
        config = readConfig(config_filename)

        # ----- DeepSeg3D prediction -----