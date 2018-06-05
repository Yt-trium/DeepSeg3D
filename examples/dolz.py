# ------------------------------------------------------------ #
#
# file : examples/dolz.py
# author : CM
# Example of training on Bullitt dataset using Jose Dolz FCNN
# 3D fully convolutional networks for subcortical
# segmentation in MRI: A large-scale study
#
# ------------------------------------------------------------ #
import os
import sys

from models.dolz import dolz_1
from utils.config.read import readConfig

from keras import backend as K

K.set_image_dim_ordering("tf")

config_filename = sys.argv[1]
if(not os.path.isfile(config_filename)):
    sys.exit(1)

config = readConfig(config_filename)

model = dolz_1(config["patch_size_x"],config["patch_size_y"],config["patch_size_z"], 2)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

print(model.input_shape)
print(model.output_shape)
model.summary()