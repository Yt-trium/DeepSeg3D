# ------------------------------------------------------------ #
#
# file : test/test_memory.py
# author : CM
# Tests for memory usage
#
# ------------------------------------------------------------ #
import time
from utils.io.read import niiToNp

image1 = niiToNp("../Datasets/Dataset/test_Images/Normal079-MRA.nii")
print(image1.nbytes)
time.sleep(10)
del image1

image2 = niiToNp("../Datasets/Dataset_pool2/test_Images/Normal079-MRA_max.nii")
print(image2.nbytes)
time.sleep(10)
del image2