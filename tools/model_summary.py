# ------------------------------------------------------------ #
#
# file : tools/model_summary.py
# author : CM
# Get information from model file
#
# ------------------------------------------------------------ #
import os
import sys

from keras import models

from utils.learning.losses import dice_loss
from utils.learning.metrics import sensitivity, specificity, precision

model_filename = sys.argv[1]
if(not os.path.isfile(model_filename)):
    sys.exit(1)

model = models.load_model(model_filename, custom_objects={'dice_loss': dice_loss,
                                                          'sensitivity': sensitivity,
                                                          'specificity': specificity,
                                                          'precision': precision})

model.summary()