# ------------------------------------------------------------ #
#
# file : test/test_normalisation.py
# author : CM
# Tests normalisation functions
#
# ------------------------------------------------------------ #
import matplotlib.pyplot as plt

from utils.config.read import readConfig
from utils.io.read import readDataset
from utils.preprocessing.normalisation import intensityNormalisationFeatureScaling, intensityMaxClipping, intensityProjection

dataset = readDataset("../Datasets/Dataset/train_Images/",
                               2,
                               448,
                               448,
                               128)

data = dataset[0]

print(data.max())
print(data.min())
print(data.mean())

plt.hist(data.flatten())
plt.show()

data = intensityMaxClipping(data, 0.5, data.dtype)

plt.hist(data.flatten())
plt.show()

data = intensityNormalisationFeatureScaling(data, data.dtype)

plt.hist(data.flatten())
plt.show()