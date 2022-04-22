import os
import glob
import h5py
import scipy.misc as misc
import numpy as np

dataset_dir = "DIV2K/"
dataset_type = "train"

f = h5py.File("DIV2K_{}x8.h5".format(dataset_type), "w")
dt = h5py.special_dtype(vlen=np.dtype('uint8'))

for subdir in ["HR", "X8"]:
    if subdir in ["HR"]:
        im_paths = glob.glob(os.path.join(dataset_dir, 
                                          "DIV2K_{}_HRX8".format(dataset_type), 
                                          "*.png"))

    else:
        im_paths = glob.glob(os.path.join(dataset_dir, 
                                          "DIV2K_{}_LR_bicubicX8".format(dataset_type),  "*.png"))
    im_paths.sort()
    grp = f.create_group(subdir)

    for i, path in enumerate(im_paths):
        im = misc.imread(path)
        print(path)
        grp.create_dataset(str(i), data=im)
