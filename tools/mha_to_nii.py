# ------------------------------------------------------------ #
#
# file : tools/mha_to_nii.py
# author : CM
# Convert mha file to nii
#
# ------------------------------------------------------------ #

import sys
import os

import SimpleITK as sitk

# input and output folder
input_folder  = sys.argv[1]
output_folder = sys.argv[2]
# input suffix
input_suffix = sys.argv[3]

if not os.path.isdir(input_folder):
    sys.exit("invalid argument")
if not os.path.isdir(output_folder):
    sys.exit("invalid argument")

# Browse all files from all directories, subdirectories...
for path, subdirs, files in os.walk(input_folder):
    for name in files:
        # Only convert files with correct suffix
        if name.endswith(input_suffix):
            # Use SimpleITK to convert file
            input_file  = os.path.join(path, name)
            output_file = os.path.join(output_folder, name[:-len(input_suffix)] + '.nii')
            image = sitk.ReadImage(input_file)
            sitk.WriteImage(image, output_file)

            print(input_file, "->", output_file, '✓')
