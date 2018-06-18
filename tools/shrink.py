# ------------------------------------------------------------ #
#
# file : tools/shrink.py
# author : CM
# Resize NIFTI files with max pooling
# Based on : github.com/Yt-trium/nii-pooling
#
# ------------------------------------------------------------ #
import os
import sys
import numpy as np
import nibabel as nib

if(not os.path.isdir(sys.argv[1])):
    sys.exit(1)

def recbrowse(src, dst):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            recbrowse(s, d)
        else:
            shrink(s,d)

def shrink(src, dst):
    print(src, "-/", pooling_kernel, "/>", dst)

    nii_src = nib.load(src)
    data_src = nii_src.get_data()
    data_dst = np.zeros(([int(x / pooling_kernel) for x in data_src.shape]), dtype=data_src.dtype)

    print(data_src.shape, "->", data_dst.shape)

    for x in range(0, data_dst.shape[0]):
        for y in range(0, data_dst.shape[1]):
            for z in range(0, data_dst.shape[2]):
                ix = int(x * pooling_kernel)
                iy = int(y * pooling_kernel)
                iz = int(z * pooling_kernel)

                pool = data_src[ix:ix + pooling_kernel, iy:iy + pooling_kernel, iz:iz + pooling_kernel]
                data_dst[x, y, z] = pool.max()

    img = nib.Nifti1Image(data_dst, nii_src.affine)
    img.to_filename(dst)

pooling_kernel = int(sys.argv[3])
recbrowse(sys.argv[1], sys.argv[2])