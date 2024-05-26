"""
@author: Yang Li
created at 2020-07-22
@contact: liyang@mi.t.u-tokyo.ac.jp
"""


import os, sys, struct
import scipy.io as sio
import numpy as np
import torch
import random
import math
import plyfile

import  sys, os
sys.path.append("../")
import data_util
import marching_cubes.marching_cubes as mc





if __name__ == '__main__':


    folder = "/home/mil/liyang/dataset/Dying/tsdf"

    for i in range (22, 32):

        view_id = "00000_10_%03d" % i


        view_sdf = os.path.join(folder, view_id + "_tsdf.bin" )


        inputs, dims, world2grid = data_util.load_scene(view_sdf)
        locs, sdf = inputs
        dimz, dimy, dimx = dims


        pred_sdf_dense = data_util.sparse_to_dense_np(
            locs, sdf[:, np.newaxis], dims[2], dims[1], dims[0], -float('inf'))


        mc.marching_cubes(torch.from_numpy(pred_sdf_dense), None, isovalue=0, truncation=4, thresh=5,
                          output_filename= view_id + ".ply")



        # data_util.visualize_sparse_sdf_as_cubes(src_locs, "src.ply")
        # data_util.visualize_sparse_sdf_as_cubes(tgt_locs, "tgt.ply")