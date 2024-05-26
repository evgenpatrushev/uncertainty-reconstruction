
import sys
sys.path.append("../")

import data_organization.tsdf_volume as fusion
import numpy as np
import cv2
import os

import  data_util
# import marching_cubes.marching_cubes as mc
import torch
import struct





def dump_scene_npz (locs, sdf, fname) :

    np.savez_compressed(fname,
                        dims=np.array([256,256,256]),
                        input_locs=locs,
                        input_sdf=sdf)

def partition_arg_topK(matrix, K, axis=0):
    """ find index of K smallest entries along a axis
    perform topK based on np.argpartition
    :param matrix: to be sorted
    :param K: select and sort the top K items
    :param axis: 0 or 1. dimension to be sorted.
    :return:
    """
    a_part = np.argpartition(matrix, K, axis=axis)
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        a_sec_argsort_K = np.argsort(matrix[a_part[0:K, :], row_index], axis=axis)
        return a_part[0:K, :][a_sec_argsort_K, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        a_sec_argsort_K = np.argsort(matrix[column_index, a_part[:, 0:K]], axis=axis)
        return a_part[:, 0:K][column_index, a_sec_argsort_K]

def knn_point_np(k, pos1, pos2):
    '''
    :param k: number of k in k-nn search
    :param pos1: (N, 3) float32 array, input points
    :param pos2: (M, 3) float32 array, query points
    :return:
    '''
    '''
    Output:
        val: (batch_size, npoint, k) float32 array, L2 distances
        idx: (batch_size, npoint, k) int32 array, indices to input points
    '''

    N, _ = pos1.shape
    M, _ = pos2.shape

    pos1 = pos1.reshape(1,N,-1).repeat(M, axis=0)
    pos2 = pos2.reshape(M,1,-1).repeat(N, axis=1)
    dist = np.sum((pos1-pos2)**2,-1)
    idx = partition_arg_topK(dist, K=k, axis=1)
    val = np.take_along_axis ( dist , idx, axis=1)
    return np.sqrt(val), idx


def voxelize_scene_flow (  flow, flow_loc, voxel_loc , knn=3 ) :
    ''' blend flow motion to voxel positions
    :param flow: [n,3]
    :param flow_loc:  [n,3]
    :param voxel_loc: [m,3]
    :return:
    '''

    dists, idx = knn_point_np (knn, flow_loc, voxel_loc )
    dists[dists < 1e-10] = 1e-10
    weight = 1.0 / dists
    weight = weight / np.sum(weight, -1, keepdims=True)  # [B,N,3]
    blended_flow = np.sum ( flow [idx] * weight.reshape ( [-1, knn, 1]), axis=1, keepdims=False )

    return blended_flow


def tri_interpolate_volume_motion (dense_motion, query):
    '''   tri-linear interpolation 8 neibs
    :param dense_motion: [dx, dy, dz, 3]
    :param query:  [N, 3]
    :return:
    '''

    dx, dy, dz, _ = dense_motion.shape

    x_indices = query[:,0:1]
    y_indices = query[:,1:2]
    z_indices = query[:,2:]

    x0 = x_indices.astype(np.integer)
    y0 = y_indices.astype(np.integer)
    z0 = z_indices.astype(np.integer)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    x = x_indices - x0
    y = y_indices - y0
    z = z_indices - z0

    _1_x = 1-x
    _1_y = 1-y
    _1_z = 1-z


    eight= {}

    eight["_000"] = dense_motion[x0[:,0], y0[:,0], z0[:,0], :] * _1_x * _1_y * _1_z
    eight["_100"] = dense_motion[x1[:,0], y0[:,0], z0[:,0], :] * x * _1_y * _1_z
    eight["_010"] = dense_motion[x0[:,0], y1[:,0], z0[:,0], :] * _1_x * y * _1_z
    eight["_001"] = dense_motion[x0[:,0], y0[:,0], z1[:,0], :] * _1_x * _1_y * z
    eight["_101"] = dense_motion[x1[:,0], y0[:,0], z1[:,0], :] * x * _1_y * z
    eight["_011"] = dense_motion[x0[:,0], y1[:,0], z1[:,0], :] * _1_x * y * z
    eight["_110"] = dense_motion[x1[:,0], y1[:,0], z0[:,0], :] * x * y * _1_z
    eight["_111"] = dense_motion[x1[:,0], y1[:,0], z1[:,0], :] * x * y * z


    intmot = np.zeros ( query.shape).astype(float)

    for k, val in eight.items():
        intmot = intmot + val

    return intmot
    # intmot = (dense_motion[x0, y0, z0, :] * (1 - x) * (1 - y) * (1 - z) +
    #           dense_motion[x1, y0, z0, :] * x * (1 - y) * (1 - z) +
    #           dense_motion[x0, y1, z0, :] * (1 - x) * y * (1 - z) +
    #           dense_motion[x0, y0, z1, :] * (1 - x) * (1 - y) * z +
    #           dense_motion[x1, y0, z1, :] * x * (1 - y) * z +
    #           dense_motion[x0, y1, z1, :] * (1 - x) * y * z +
    #           dense_motion[x1, y1, z0, :] * x * y * (1 - z) +
    #           dense_motion[x1, y1, z1, :] * x * y * z)

def tri_interpolate_volume_occ (dense_occ, query):
    '''   tri-linear interpolation 8 neibs
    :param dense_motion: [dx, dy, dz]
    :param query:  [N, 3]
    :return:
    '''

    dx, dy, dz = dense_occ.shape

    x_indices = query[:,0]
    y_indices = query[:,1]
    z_indices = query[:,2]

    x0 = x_indices.astype(np.integer)
    y0 = y_indices.astype(np.integer)
    z0 = z_indices.astype(np.integer)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1


    eight= {}

    eight["_000"] = dense_occ[x0 , y0 , z0   ]
    eight["_100"] = dense_occ[x1 , y0 , z0   ]
    eight["_010"] = dense_occ[x0 , y1 , z0   ]
    eight["_001"] = dense_occ[x0 , y0 , z1   ]
    eight["_101"] = dense_occ[x1 , y0 , z1   ]
    eight["_011"] = dense_occ[x0 , y1 , z1   ]
    eight["_110"] = dense_occ[x1 , y1 , z0   ]
    eight["_111"] = dense_occ[x1 , y1 , z1   ]


    occ = np.zeros ( query.shape[0]).astype(float)

    for k, val in eight.items():
        occ = occ  +  val

    return occ


def depth2sdf ( depth_im, cam_intr , voxel_size, trunc = 3) :

    H, W = depth_im.shape

    cam_pose = np.eye(4)
    vol_bnds = np.zeros((3, 2))

    # Compute camera view frustum and extend convex hull
    view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
    vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
    vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))

    Volume = fusion.TSDFVolume(vol_bnds, voxel_size=voxel_size, trunc=trunc)

    color = np.ones([H, W, 3]) * 255

    Volume.integrate(color, depth_im, cam_intr, cam_pose, obs_weight=1.)

    """tsdf is [-1,1]"""
    tsdf_vol, color_vol = Volume.get_volume()

    '''adjust axis'''
    # tsdf_vol = np.flip(tsdf_vol, 1)  # x-y-z to z-y-x

    locs, sdf = data_util.dense_to_sparse_np (tsdf_vol, 1 )

    # vol_origin = tsdf_vol.get_vol_origin()

    sdf = sdf * trunc
    # locs = np.stack( [locs[1], locs[2],locs[0]])  # z-y-x to y-x-z
    # locs = np.stack( [locs[0], locs[1],locs[2]])  # z-y-x to y-x-z

    # dim = np.array([ int(locs[0].max()*1.5), int(locs[1].max()*2.5), int(locs[2].max()*1.5) ])
    dim = np.array([locs[0].max()*1.2, locs[1].max()*1.2, locs[2].max() * 1.5])
    # dim = np.array([locs[0].max()*1.1, locs[1].max()*1.1, locs[2].max() * 1.5])

    # print (sdf.shape)
    # print (locs.shape)
    # print( dim)
    return  locs, sdf, dim, Volume._vol_origin





if __name__ == '__main__':

    # folder = "/home/mil/liyang/dataset/depth"
    '''Azure kinect'''
    # cam_intr = np.array([443.405000, 0.0, 256.000000,
    #                      0.0, 443.405000, 256.000000, 0,0,0]).reshape([3,3])

    fname = "./depth/minions.png"
    '''VolumeDeform sequence'''
    cam_intr = np.array([570.342000, 0.0, 320.000000,
                         0.0, 570.405000, 240.000000, 0,0,1]).reshape([3,3])

    voxel_size = 0.007
    trunc = 3


    depth_im = cv2.imread( fname, -1).astype(float)
    depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
    depth_im[depth_im > 10] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)


    locs, sdf, dimension = depth2sdf( depth_im, cam_intr, voxel_size )

    a = 1

