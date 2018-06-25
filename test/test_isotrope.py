# ------------------------------------------------------------ #
#
# file : test/test_isotrope.py
# author : CM
# Tests to convert image to the same size and make them isotrope
#
# ------------------------------------------------------------ #
import os
import nibabel as nib

from dipy.align.reslice import reslice
from dipy.data import get_data

folder = "../Datasets/VIVABRAIN_LIGHT/"

files = os.listdir(folder)
files.sort()

image = nib.load(os.path.join(folder, files[0]))
data  = image.get_data()

# print(image)
print(data.shape)
print(image.affine)
print(image.header.get_zooms()[:3])

affine = image.affine
zooms = image.header.get_zooms()[:3]
new_zooms = (1., 1., 1.)

data2, affine2 = reslice(data, affine, zooms, new_zooms)
print(data2.shape)

img2 = nib.Nifti1Image(data2, affine2)
nib.save(img2, os.path.join(folder, files[0]+"toto.nii.gz"))

# Todo : Test way to make isotrope image from non-isotrope and resize all image to the same size