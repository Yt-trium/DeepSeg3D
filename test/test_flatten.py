# ------------------------------------------------------------ #
#
# file : test/test_flatten.py
# author : CM
# Tests for Dolz model output
#
# ------------------------------------------------------------ #

import numpy as np

from keras.utils import to_categorical

test1 = np.random.randint(0,2,size=(16,16,16))

print(test1.shape)

test2 = to_categorical(test1.flatten(),2)

print(test2.shape)

pred = np.argmax(test2, axis=1)

print(pred.shape)

pred = pred.reshape(16,16,16)

print(np.array_equal(test1,pred))