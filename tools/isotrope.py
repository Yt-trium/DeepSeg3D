# ------------------------------------------------------------ #
#
# file : tools/isotrope.py
# author : CM
# Make a dataset isotrope
#
# ------------------------------------------------------------ #

import sys
import os
import nibabel as nib
from dipy.align.reslice import reslice

# input and output folder
input_folder  = sys.argv[1]
output_folder = sys.argv[2]

if not os.path.isdir(input_folder):
    sys.exit("invalid argument")
if not os.path.isdir(output_folder):
    sys.exit("invalid argument")

input_suffix = ".nii"

# Browse all files from all directories, subdirectories...
for path, subdirs, files in os.walk(input_folder):
    files.sort()
    for name in files:
        if name.endswith(input_suffix):
            input_file  = os.path.join(path, name)
            output_file = input_file.replace(input_folder, output_folder)

            image = nib.load(input_file)

            data = image.get_data()
            affine = image.affine
            zooms = image.header.get_zooms()[:3]
            # new_zooms = (min(zooms), min(zooms), min(zooms))
            new_zooms = (zooms[0]*2, zooms[0]*2, zooms[0]*2+0.04)

            data_iso, affine_iso = reslice(data, affine, zooms, new_zooms)

            img = nib.Nifti1Image(data_iso, affine_iso)

            img.to_filename(output_file)

            print(input_file, "->", output_file, '✓')