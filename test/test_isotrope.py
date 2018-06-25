# ------------------------------------------------------------ #
#
# file : test/test_isotrope.py
# author : CM
# Tests to convert image to the same size and make them isotrope
#
# ------------------------------------------------------------ #
import os
import nibabel as nib

files = os.listdir("../Datasets/Dataset/train_Images/")
files.sort()

image = nib.load(os.path.join("../Datasets/Dataset/train_Images/", files[0]))
data  = image.get_data()

print(image)
print(data.shape)
print(image.affine)

# Todo : Test way to make isotrope image from non-isotrope and resize all image to the same size