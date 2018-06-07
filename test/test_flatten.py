# ------------------------------------------------------------ #
#
# file : test/test_flatten.py
# author : CM
# Tests for Dolz model output
#
# ------------------------------------------------------------ #
from random import randint

import numpy as np

from keras.utils import to_categorical

# Random image
from utils.learning.patch.extraction import extractPatch, generateFullPatchsCentered
from utils.learning.patch.reconstruction import dolzReconstruction

input = np.random.randint(0,2,size=(2,224,224,64))
print("dataset shape", input.shape)
print("input image shape", input[0].shape)

patchs = generateFullPatchsCentered(input[0],32,32,32)

print(patchs.max())
print(patchs.min())

patchs_center = np.empty((784,16,16,16))
for i in range(0,patchs.shape[0]):
    #patchs_center[i] = patchs[i,7:23,7:23,7:23]
    patchs_center[i] = patchs[i,8:24,8:24,8:24]

patchs_flat = np.empty((784,4096,2))
for i in range(0, patchs_center.shape[0]):
    patchs_flat[i] = to_categorical(patchs_center[i].flatten(),2)

print("patchs shape", patchs.shape)
print("patchs center shape", patchs_center.shape)
print("patchs flat shape", patchs_flat.shape)

prediction = np.argmax(patchs_flat, axis=2)

print("prediction shape", prediction.shape)

patchs_reconstruct = np.empty((784,16,16,16))
for i in range(0,prediction.shape[0]):
    patchs_reconstruct[i] = prediction[i].reshape(16,16,16)

print("patchs reconstruct shape", patchs_reconstruct.shape)

output = dolzReconstruction(input[0],patchs_flat)

print("output shape", output.shape)

print(np.array_equal(input[0],output))
