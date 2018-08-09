# ------------------------------------------------------------ #
#
# file : tools/dataset_statistics.py
# author : CM
# Read a full dataset to get some stats about it
#
# ------------------------------------------------------------ #

import sys
import os

import numpy as np
import nibabel as nib

# input and output folder
in_folder  = sys.argv[1]
gd_folder = sys.argv[2]

if not os.path.isdir(in_folder):
    sys.exit("invalid argument")
if not os.path.isdir(gd_folder):
    sys.exit("invalid argument")

in_files = os.listdir(in_folder)
in_files.sort()

gd_files = os.listdir(gd_folder)
gd_files.sort()

global_in_max = 0
global_in_min = float("inf")
global_in_mean = []

global_gd_max = 0
global_gd_min = float("inf")
global_gd_mean = []

for i in range(len(in_files)):
    print(i+1, '/', len(in_files), in_files[i])

    if not in_files[i] == gd_files[i]:
        sys.exit("files not matching")

    in_image = nib.load(os.path.join(in_folder, in_files[i]))
    gd_image = nib.load(os.path.join(gd_folder, gd_files[i]))

    if not (in_image.affine == gd_image.affine).all():
        sys.exit("affine not matching")

    in_data = in_image.get_data()
    gd_data = gd_image.get_data()

    if not in_data.shape == gd_data.shape:
        sys.exit("shape not matching")

    in_data = in_data.flatten()
    gd_data = gd_data.flatten()

    print("IN MAX : ", in_data.max())
    print("IN MIN : ", in_data.min())
    print("IN AVG : ", in_data.mean())

    print("GD MAX : ", gd_data.max())
    print("GD MIN : ", gd_data.min())
    print("GD AVG : ", gd_data.mean())

    global_in_max = max(global_in_max, in_data.max())
    global_in_min = min(global_in_min, in_data.min())
    global_in_mean.append(in_data.mean())

    global_gd_max = max(global_gd_max, gd_data.max())
    global_gd_min = min(global_gd_min, gd_data.min())
    global_gd_mean.append(gd_data.mean())

print("GLOBAL IN MAX : ", global_in_max)
print("GLOBAL IN MIN : ", global_in_min)
print("GLOBAL IN AVG : ", np.array(global_in_mean).mean())

print("GLOBAL GD MAX : ", global_gd_max)
print("GLOBAL GD MIN : ", global_gd_min)
print("GLOBAL GD AVG : ", np.array(global_gd_mean).mean())