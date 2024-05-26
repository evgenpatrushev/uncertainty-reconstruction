import os, sys, struct
import scipy.io as sio
import numpy as np
import torch
import random
import math
import plyfile

from plyfile import (PlyData, PlyElement, make2d,
                     PlyHeaderParseError, PlyElementParseError,
                     PlyProperty)

import marching_cubes.marching_cubes as mc


def get_train_files(data_path, file_list, val_file_list):
    names = open(file_list).read().splitlines()
    if not '.' in names[0]:
        names = [name + '__0__.sdf' for name in names]
    files = [os.path.join(data_path, f) for f in names]
    val_files = []
    if val_file_list:
        val_names = open(val_file_list).read().splitlines()
        val_files = [os.path.join(data_path, f) for f in val_names]
    return files, val_files


def dump_args_txt(args, output_file):
    with open(output_file, 'w') as f:
        f.write('%s\n' % str(args))


# locs: hierarchy, then batches
def compute_batchids(output_occs, output_sdf, batch_size):
    batchids = [None] * (len(output_occs) + 1)
    for h in range(len(output_occs)):
        batchids[h] = [None] * batch_size
        for b in range(batch_size):
            batchids[h][b] = output_occs[h][0][:, -1] == b
    batchids[-1] = [None] * batch_size
    for b in range(batch_size):
        batchids[-1][b] = output_sdf[0][:, -1] == b
    return batchids


# locs: zyx ordering
def sparse_to_dense_np(locs, values, dimx, dimy, dimz, default_val):
    nf_values = 1 if len(values.shape) == 1 else values.shape[1]
    dense = np.zeros([dimz, dimy, dimx, nf_values], dtype=values.dtype)
    dense.fill(default_val)
    # print('dense', dense.shape)
    # print('locs', locs.shape)
    # print('values', values.shape)
    dense[locs[:, 0], locs[:, 1], locs[:, 2], :] = values
    if nf_values > 1:
        return dense.reshape([dimz, dimy, dimx, nf_values])
    return dense.reshape([dimz, dimy, dimx])


def sparse_to_dense_np_xyz(locs, values, dimx, dimy, dimz, default_val):
    nf_values = 1 if len(values.shape) == 1 else values.shape[1]
    dense = np.zeros([dimx, dimy, dimz, nf_values], dtype=values.dtype)
    dense.fill(default_val)
    # print('dense', dense.shape)
    # print('locs', locs.shape)
    # print('values', values.shape)
    dense[locs[:, 0], locs[:, 1], locs[:, 2], :] = values
    if nf_values > 1:
        return dense.reshape([dimx, dimy, dimz, nf_values])
    return dense.reshape([dimx, dimy, dimz])


def dense_to_sparse_np(grid, thresh):
    locs = np.where(np.abs(grid) < thresh)
    values = grid[locs[0], locs[1], locs[2]]
    locs = np.stack(locs)
    return locs, values


def load_train_npz(file):
    voxel_size = 0.01
    with open(file, 'rb') as fp:
        data = np.load(fp)

        dims = data['dims']
        dimz, dimy, dimx = dims

        input_locs = data['input_locs'].transpose()
        input_sdfs = data['input_sdf']
        input_sdfs /= voxel_size

        target_sdfs = data['tgt_dense']
        target_sdfs /= voxel_size

        h0_dense = data['h0_dense'] / voxel_size  # 4

        h1_dense = data['h1_dense'] / voxel_size  # 2

        h2_dense = data['h2_dense'] / voxel_size  # 1

        world2grid = np.eye(4)

        hierarchy = [h2_dense, h1_dense, h0_dense]

        return [input_locs, input_sdfs], target_sdfs, [dimz, dimy, dimx], world2grid, hierarchy


def load_train_motion_shape_complete(file, voxel_size=0.02):
    with open(file, 'rb') as fp:
        data = np.load(fp)

        dims = data['dims']
        dimx, dimy, dimz = dims

        input_locs = data['input_locs'].transpose()
        input_motion = data['input_motion']
        input_motion /= voxel_size

        target_locs = data['target_locs'].transpose()
        target_motion = data['tgt_motion']
        target_motion /= voxel_size

        h0_mot_dense = data['h0_mot_dense'] / voxel_size
        h1_mot_dense = data['h1_mot_dense'] / voxel_size
        # h2_mot_dense = data['h2_mot_dense'] / voxel_size
        # mot_hierarchy = [h2_mot_dense, h1_mot_dense, h0_mot_dense]
        mot_hierarchy = [h1_mot_dense, h0_mot_dense]

        input_sdfs = data['input_sdf']
        input_sdfs /= voxel_size

        target_sdfs = data['tgt_sdf_dense']
        target_sdfs /= voxel_size

        h0_sdf_dense = data['h0_sdf_dense'] / voxel_size  # 4
        h1_sdf_dense = data['h1_sdf_dense'] / voxel_size  # 2
        # h2_sdf_dense = data['h2_sdf_dense'] / voxel_size  # 2
        # sdf_hierarchy = [h2_sdf_dense, h1_sdf_dense, h0_sdf_dense]
        sdf_hierarchy = [h1_sdf_dense, h0_sdf_dense]

        return [input_locs, input_motion], [target_locs, target_motion], [dimx, dimy, dimz], \
               input_sdfs, target_sdfs, sdf_hierarchy, mot_hierarchy


def load_train_file(file):
    '''load tsdfs from raw data, load data for all hirachies'''
    fin = open(file, 'rb')
    dimx = struct.unpack('Q', fin.read(8))[0]
    dimy = struct.unpack('Q', fin.read(8))[0]
    dimz = struct.unpack('Q', fin.read(8))[0]
    voxelsize = struct.unpack('f', fin.read(4))[0]
    world2grid = struct.unpack('f' * 4 * 4, fin.read(4 * 4 * 4))
    world2grid = np.asarray(world2grid, dtype=np.float32).reshape([4, 4])
    # input data
    num = struct.unpack('Q', fin.read(8))[0]
    input_locs = struct.unpack('I' * num * 3, fin.read(num * 3 * 4))
    input_locs = np.asarray(input_locs, dtype=np.int32).reshape([num, 3])
    input_locs = np.flip(input_locs, 1).copy()  # convert to zyx ordering
    input_sdfs = struct.unpack('f' * num, fin.read(num * 4))
    input_sdfs = np.asarray(input_sdfs, dtype=np.float32)
    input_sdfs /= voxelsize
    # target data
    num = struct.unpack('Q', fin.read(8))[0]
    target_locs = struct.unpack('I' * num * 3, fin.read(num * 3 * 4))
    target_locs = np.asarray(target_locs, dtype=np.int32).reshape([num, 3])
    target_locs = np.flip(target_locs, 1).copy()  # convert to zyx ordering
    target_sdfs = struct.unpack('f' * num, fin.read(num * 4))
    target_sdfs = np.asarray(target_sdfs, dtype=np.float32)
    target_sdfs /= voxelsize
    target_sdfs = sparse_to_dense_np(target_locs, target_sdfs[:, np.newaxis], dimx, dimy, dimz, -float('inf'))
    # known data
    num = struct.unpack('Q', fin.read(8))[0]
    assert (num == dimx * dimy * dimz)
    target_known = struct.unpack('B' * dimz * dimy * dimx, fin.read(dimz * dimy * dimx))
    target_known = np.asarray(target_known, dtype=np.uint8).reshape([dimz, dimy, dimx])
    # pre-computed hierarchy
    hierarchy = []
    factor = 2
    for h in range(3):
        num = struct.unpack('Q', fin.read(8))[0]
        hlocs = struct.unpack('I' * num * 3, fin.read(num * 3 * 4))
        hlocs = np.asarray(hlocs, dtype=np.int32).reshape([num, 3])
        hlocs = np.flip(hlocs, 1).copy()  # convert to zyx ordering
        hvals = struct.unpack('f' * num, fin.read(num * 4))
        hvals = np.asarray(hvals, dtype=np.float32)
        hvals /= voxelsize
        grid = sparse_to_dense_np(hlocs, hvals[:, np.newaxis], dimx // factor, dimy // factor, dimz // factor,
                                  -float('inf'))
        hierarchy.append(grid)
        factor *= 2
    hierarchy.reverse()
    return [input_locs, input_sdfs], target_sdfs, [dimz, dimy, dimx], world2grid, target_known, hierarchy


def load_scene(file, is_npz=False):
    if not is_npz:
        fin = open(file, 'rb')
        dimx = struct.unpack('Q', fin.read(8))[0]
        dimy = struct.unpack('Q', fin.read(8))[0]
        dimz = struct.unpack('Q', fin.read(8))[0]
        voxelsize = struct.unpack('f', fin.read(4))[0]
        world2grid = struct.unpack('f' * 4 * 4, fin.read(4 * 4 * 4))
        world2grid = np.asarray(world2grid, dtype=np.float32).reshape([4, 4])
        # data
        num = struct.unpack('Q', fin.read(8))[0]
        locs = struct.unpack('I' * num * 3, fin.read(num * 3 * 4))
        locs = np.asarray(locs, dtype=np.int32).reshape([num, 3])
        locs = np.flip(locs, 1).copy()  # convert to zyx ordering
        sdf = struct.unpack('f' * num, fin.read(num * 4))
        sdf = np.asarray(sdf, dtype=np.float32)
        sdf /= voxelsize
        fin.close()
        return [locs, sdf], [dimz, dimy, dimx], world2grid

    else:
        with open(file, 'rb') as fp:
            data = np.load(fp)
            dims = data['dims']
            dimz, dimy, dimx = dims

            locs = data['input_locs'].transpose()
            sdf = data['input_sdf']
            world2grid = np.eye(4)

            return [locs, sdf], [dimz, dimy, dimx], world2grid


# def load_scene_known(file):
#     #assert os.path.isfile(file)
#     fin = open(file, 'rb')
#     dimx = struct.unpack('Q', fin.read(8))[0]
#     dimy = struct.unpack('Q', fin.read(8))[0]
#     dimz = struct.unpack('Q', fin.read(8))[0]
#     voxelsize = struct.unpack('f', fin.read(4))[0]
#     world2grid = struct.unpack('f'*4*4, fin.read(4*4*4))
#     world2grid = np.asarray(world2grid, dtype=np.float32).reshape([4, 4])
#     known = struct.unpack('B'*dimz*dimy*dimx, fin.read(dimz*dimy*dimx))
#     known = np.asarray(known, dtype=np.uint8).reshape([dimz, dimy, dimx])
#     fin.close()
#     return known


def preprocess_sdf_np(sdf, truncation):
    sdf[sdf < -truncation] = -truncation
    sdf[sdf > truncation] = truncation
    return sdf


def preprocess_sdf_pt(sdf, truncation):
    # sdf to tsdf
    sdf[sdf < -truncation] = -truncation
    sdf[sdf > truncation] = truncation
    return sdf


def unprocess_sdf_pt(sdf, truncation):
    return sdf


def visualize_sdf_as_points(sdf, iso, output_file, transform=None):
    # collect verts from sdf
    verts = []
    for z in range(sdf.shape[0]):
        for y in range(sdf.shape[1]):
            for x in range(sdf.shape[2]):
                if abs(sdf[z, y, x]) < iso:
                    verts.append(np.array([x, y, z]) + 0.5)  # center of voxel
    if len(verts) == 0:
        print('warning: no valid sdf points for %s' % output_file)
        return
    verts = np.stack(verts)
    visualize_points(verts, output_file, transform)


def visualize_sparse_sdf_as_points(sdf_locs, sdf_vals, iso, output_file, transform=None):
    # collect verts from sdf
    mask = np.abs(sdf_vals) < iso
    verts = sdf_locs[:, :3][mask]
    if len(verts) == 0:
        print('warning: no valid sdf points for %s' % output_file)
        return
    verts = np.stack(verts).astype(np.float32)
    verts = verts[:, ::-1] + 0.5
    visualize_points(verts, output_file, transform)


# def visualize_sparse_sdf_as_voxels (sdf_locs, sdf_vals, iso, output_file, transform=None)

def visualize_sparse_sdf_as_cubes(sdf_locs, output_file, factor=1, transform=None):
    # collect verts from sdf

    verts = sdf_locs[:, :3]

    if len(verts) == 0:
        print('warning: no valid sdf points for %s' % output_file)
        return
    verts = np.stack(verts).astype(np.float32)
    # verts = verts[:, ::-1] + 0.5

    n_cube = verts.shape[0]

    verts = np.expand_dims(verts, axis=1).repeat(8, axis=1)

    offset = np.array(
        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 1],
         [1, 0, 1],
         [0, 1, 0],
         [1, 1, 0],
         [0, 1, 1],
         [1, 1, 1]])[np.newaxis, :] - 0.5
    # * 0.8 * factor
    offset = offset * 0.9

    # offset = np.stack(np.nonzero( np.ones([2,2,2]))).transpose()[np.newaxis,:] * 0.8

    verts = verts + offset
    verts = verts.reshape([-1, 3])

    verts = np.array([tuple(v) for v in verts], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    verts = plyfile.PlyElement.describe(verts, 'vertex')

    faces = np.array([[4, 7, 5],
                      [4, 6, 7],
                      [0, 2, 4],
                      [2, 6, 4],
                      [0, 1, 2],
                      [1, 3, 2],
                      [1, 5, 7],
                      [1, 7, 3],
                      [2, 3, 7],
                      [2, 7, 6],
                      [0, 4, 1],
                      [1, 4, 5]])

    faces = faces.reshape([1, -1, 3]).repeat(n_cube, axis=0)  # [n,11,3]
    # vert_offset = np.arange()
    vert_offset = np.arange(n_cube).reshape([-1, 1, 1]) * 8  # [n,1, 1]
    faces = faces + vert_offset
    faces = faces.reshape([-1, 3])

    faces_array = np.empty(len(faces), dtype=[('vertex_indices', 'i4', (3,))])
    faces_array['vertex_indices'] = faces
    PLY_faces = PlyElement.describe(faces_array, 'face')

    plyfile.PlyData([verts, PLY_faces]).write(output_file)


def visualize_occ_as_points(sdf, thresh, output_file, transform=None, thresh_max=float('inf')):
    # collect verts from sdf
    verts = []
    for z in range(sdf.shape[0]):
        for y in range(sdf.shape[1]):
            for x in range(sdf.shape[2]):
                val = abs(sdf[z, y, x])
                if val > thresh and val < thresh_max:
                    verts.append(np.array([x, y, z]) + 0.5)  # center of voxel
    if len(verts) == 0:
        print('warning: no valid occ points for %s' % output_file)
        return
    # print('[visualize_occ_as_points]', output_file, len(verts))
    verts = np.stack(verts)
    visualize_points(verts, output_file, transform)


def visualize_sparse_locs_as_points(locs, output_file, transform=None):
    # collect verts from sdf
    verts = locs[:, :3]
    if len(verts) == 0:
        print('warning: no valid occ points for %s' % output_file)
        return
    # print('[visualize_occ_as_points]', output_file, len(verts))
    verts = np.stack(verts).astype(np.float32)
    verts = verts[:, ::-1] + 0.5
    visualize_points(verts, output_file, transform)


def visualize_points(points, output_file, transform=None, colors=None):
    verts = points if points.shape[1] == 3 else np.transpose(points)
    if transform is not None:
        x = np.ones((verts.shape[0], 4))
        x[:, :3] = verts
        x = np.matmul(transform, np.transpose(x))
        x = np.transpose(x)
        verts = np.divide(x[:, :3], x[:, 3, None])

    ext = os.path.splitext(output_file)[1]
    if colors is not None:
        colors = np.clip(colors, 0, 1)
    if colors is not None or ext == '.obj':
        output_file = os.path.splitext(output_file)[0] + '.obj'
        num_verts = len(verts)
        with open(output_file, 'w') as f:
            for i in range(num_verts):
                v = verts[i]
                if colors is None:
                    f.write('v %f %f %f\n' % (v[0], v[1], v[2]))
                else:
                    f.write('v %f %f %f %f %f %f\n' % (v[0], v[1], v[2], colors[i, 0], colors[i, 1], colors[i, 2]))
    elif ext == '.ply':
        verts = np.array([tuple(v) for v in verts], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        el = plyfile.PlyElement.describe(verts, 'vertex')
        plyfile.PlyData([el]).write(output_file)
    else:
        raise NotImplementedError  # unsupported extension


def make_scale_transform(scale):
    if isinstance(scale, int) or isinstance(scale, float):
        scale = [scale, scale, scale]
    assert (len(scale) == 3)
    transform = np.eye(4, 4)
    for k in range(3):
        transform[k, k] = scale[k]
    return transform


def extract_mesh(output_fname, locs, sdf, dims, isovalue=0, trunc=3):
    dims[2], dims[1], dims[0] = dims
    sdf_dense = sparse_to_dense_np(locs, sdf[:, np.newaxis], dims[2], dims[1], dims[0], -float('inf'))
    mc.marching_cubes(torch.from_numpy(sdf_dense), None, isovalue=isovalue, truncation=trunc, thresh=10,
                      output_filename=os.path.join(output_fname))


def get_mesh(locs, sdf, dims, isovalue=0, trunc=3):
    dims[2], dims[1], dims[0] = dims
    sdf_dense = sparse_to_dense_np(locs, sdf[:, np.newaxis], dims[2], dims[1], dims[0], -float('inf'))
    # mc.marching_cubes(torch.from_numpy(sdf_dense), None, isovalue=isovalue, truncation=trunc, thresh=10,
    #                   output_filename=os.path.join(output_fname))

    vertices, vertcolors, faces = mc.run_marching_cubes(torch.from_numpy(sdf_dense), None, isovalue=isovalue,
                                                        truncation=trunc, thresh=10)

    return vertices, vertcolors, faces


def save_mesh(verts, colors, indices, output_file):
    ext = os.path.splitext(output_file)[1]
    if ext == '.obj':
        with open(output_file, 'w') as f:
            for k in range(len(verts)):
                v = verts[k]
                c = colors[k]
                f.write('v %f %f %f %d %d %d\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
            f.write('g foo\n')
            for ind in indices:
                f.write('f %d %d %d\n' % (ind[0] + 1, ind[1] + 1, ind[2] + 1))
            f.write('g\n')
    else:
        verts = np.array([tuple(v) for v in verts], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        face = np.array([(f.tolist(), 220, 220, 220) for f in indices],
                        dtype=[('vertex_indices', 'i4', (3,)), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        el_vert = plyfile.PlyElement.describe(verts, 'vertex')
        el_face = plyfile.PlyElement.describe(face, 'face')
        plyfile.PlyData([el_vert, el_face]).write(output_file)


def save_predictions(output_path, names, inputs, target_for_sdf, target_for_occs, output_sdf, output_occs, world2grids,
                     truncation, thresh=1):
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    if output_occs is not None:
        num_hierarchy_levels = len(output_occs)
        factors = [1] * num_hierarchy_levels
        for h in range(num_hierarchy_levels - 2, -1, -1):
            factors[h] = factors[h + 1] * 2
    dims = np.maximum(np.max(output_sdf[0][0], 0),
                      np.max(inputs[0], 0)) + 1 if target_for_sdf is None else target_for_sdf.shape[2:]
    isovalue = 0
    trunc = truncation - 0.1
    ext = '.ply'

    for k in range(len(names)):
        name = names[k]
        mask = inputs[0][:, -1] == k
        locs = inputs[0][mask]
        feats = inputs[1][mask]

        input = sparse_to_dense_np(locs[:, :-1], feats, dims[2], dims[1], dims[0], -float('inf'))
        mc.marching_cubes(torch.from_numpy(input), None, isovalue=isovalue, truncation=trunc, thresh=10,
                          output_filename=os.path.join(output_path, name + '-input-mesh' + ext))
        if output_occs is not None:
            for h in range(num_hierarchy_levels):
                transform = make_scale_transform(factors[h])
                if target_for_occs is not None:
                    visualize_occ_as_points(target_for_occs[h][k, 0] == 1, 0.5,
                                            os.path.join(output_path, name + 'target-' + str(h) + ext), transform,
                                            thresh_max=1.5)
                if output_occs is not None and output_occs[h][k] is not None:
                    visualize_sparse_locs_as_points(output_occs[h][k],
                                                    os.path.join(output_path, name + 'pred-' + str(h) + ext), transform)
        if output_sdf[k] is not None:
            locs = output_sdf[k][0][:, :3]
            pred_sdf_dense = sparse_to_dense_np(locs, output_sdf[k][1][:, np.newaxis], dims[2], dims[1], dims[0],
                                                -float('inf'))
            mc.marching_cubes(torch.from_numpy(pred_sdf_dense), None, isovalue=isovalue, truncation=trunc, thresh=10,
                              output_filename=os.path.join(output_path, name + '-pred-mesh' + ext))
        if target_for_sdf is not None:
            target = target_for_sdf[k, 0]
            mc.marching_cubes(torch.from_numpy(target), None, isovalue=isovalue, truncation=trunc, thresh=10,
                              output_filename=os.path.join(output_path, name + 'target-mesh' + ext))
