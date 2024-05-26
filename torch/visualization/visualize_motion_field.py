

import  sys, os



sys.path.append("../")

from visualization.mesh_util import get_motion_mesh, get_voxel_mesh, compute_3d_norm_vector_cmap

import open3d as o3d
import data_util
import struct
import numpy as np
from data_util import  sparse_to_dense_np_xyz
import torch


import marching_cubes.marching_cubes as mc
# import date_util


def load_scene(file, is_npz = False):
    if not is_npz:
        fin = open(file, 'rb')
        dimx = struct.unpack('Q', fin.read(8))[0]
        dimy = struct.unpack('Q', fin.read(8))[0]
        dimz = struct.unpack('Q', fin.read(8))[0]
        voxelsize = struct.unpack('f', fin.read(4))[0]
        world2grid = struct.unpack('f'*4*4, fin.read(4*4*4))
        world2grid = np.asarray(world2grid, dtype=np.float32).reshape([4, 4])
        # data
        num = struct.unpack('Q', fin.read(8))[0]
        locs = struct.unpack('I'*num*3, fin.read(num*3*4))
        locs = np.asarray(locs, dtype=np.int32).reshape([num, 3])
        # locs = np.flip(locs,1).copy() # convert to zyx ordering
        sdf = struct.unpack('f'*num, fin.read(num*4))
        sdf = np.asarray(sdf, dtype=np.float32)
        # sdf /= voxelsize
        fin.close()
        return [locs, sdf], [dimz, dimy, dimx], world2grid, voxelsize

    else :
        with open(file, 'rb') as fp:
            data = np.load(fp)
            dims = data['dims']
            dimz, dimy, dimx = dims

            locs = data['input_locs'].transpose()
            sdf = data['input_sdf']
            world2grid = np.eye(4)

            return [locs, sdf], [dimz, dimy, dimx], world2grid



def load_motion(file, is_npz = False):
    ''' not quantized flow
    :param file:
    :param is_npz:
    :return: flow: [num, 3]
    '''
    if not is_npz:
        fin = open(file, 'rb')
        num = struct.unpack('Q', fin.read(8))[0]
        flow = struct.unpack('f'*num*3, fin.read(num*3*4))
        flow = np.asarray(flow, dtype=np.float32).reshape([num, 3])
        # flow = np.flip(flow,1).copy() # convert to zyx ordering
        fin.close()
        return flow


if __name__ == '__main__':

    from matplotlib import cm
    cmap = cm.get_cmap('jet')



    folder = "/home/adl4cv/003_Capoeira/tsdf/"

    level = "20"
    voxelsize = float(level) * 0.001
    frame_id = "00000_"
    current_dim = 256
    camera = "_028_"


    sdf_file = os.path.join( folder, frame_id + level + "_tsdf.bin")
    inputs, dims, world2grid,  _ = load_scene(sdf_file)
    locs , sdf =  inputs
    sdf = sdf / voxelsize

    motion_file = os.path.join( folder, frame_id + level + "_flow.bin")
    flow = load_motion(motion_file)

    dense_flow = sparse_to_dense_np_xyz(locs, flow , current_dim, current_dim, current_dim, -float('inf'))


    """save src mesh"""
    mc_tgt = False
    if mc_tgt:

        dense_target = sparse_to_dense_np_xyz(
            locs, sdf.reshape([-1, 1]), current_dim, current_dim, current_dim, float('inf'))
        tsdf = torch.from_numpy(dense_target)
        vertices, vertcolors, faces = \
            mc.run_marching_cubes(tsdf, colors=None, isovalue=0, truncation=3, thresh=10)

        vertices = vertices.cpu().numpy()
        vertices = np.flip(vertices, 1)
        faces = faces.cpu().numpy()
        faces = np.flip(faces, 1)
        vertcolors = np.ones_like(vertices) * 0.75
        target_mesh = o3d.geometry.TriangleMesh()
        target_mesh.vertices = o3d.utility.Vector3dVector(vertices * voxelsize)
        target_mesh.triangles = o3d.utility.Vector3iVector(faces)
        target_mesh.vertex_colors = o3d.utility.Vector3dVector(vertcolors)
        target_mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(
            os.path.join("./meshes/", "tgt_mesh.ply" ), target_mesh)


    save_complete_voxels = True
    if save_complete_voxels :
        c_src = 1 - (sdf + 3) / 6
        color = cmap(c_src)[:, :3]
        voxel_mesh = get_voxel_mesh(locs  * voxelsize, voxelsize, factor=1, color=color)
        output_ply = "./meshes/" + "complete_sdf.ply"
        o3d.io.write_triangle_mesh(output_ply, voxel_mesh)


    save_flow_field = False
    if save_flow_field :
        mask = sdf < 1
        locs = locs[mask]
        flow = flow[mask]

        # flow = flow + np.asarray ( [ [0,-0.1,0]] )

        flow_cmap =  compute_3d_norm_vector_cmap(flow)
        flow_mesh = get_motion_mesh(locs * voxelsize, flow,
                                    flow_cmap,
                                    radius=voxelsize * 0.12)

        output_ply = "meshes/" + level + "_field.ply"
        o3d.io.write_triangle_mesh(output_ply, flow_mesh)


    """partial motion"""
    sdf_file = os.path.join( folder, frame_id + level + camera + "tsdf.bin")
    inputs, dims, world2grid,  _ = load_scene(sdf_file)
    par_locs , par_sdf =  inputs
    par_sdf =  par_sdf / voxelsize


    """save src mesh"""
    mc_src = False
    if mc_src:

        dense_target = sparse_to_dense_np_xyz(
            par_locs, par_sdf.reshape([-1, 1]), current_dim, current_dim, current_dim, float('inf'))
        tsdf = torch.from_numpy(dense_target )
        vertices, vertcolors, faces = \
            mc.run_marching_cubes(tsdf, colors=None, isovalue=0, truncation=3, thresh=10)

        vertices = vertices.cpu().numpy()
        vertices = np.flip(vertices, 1)
        faces = faces.cpu().numpy()
        faces = np.flip(faces, 1)
        vertcolors = np.ones_like(vertices) * 0.75
        target_mesh = o3d.geometry.TriangleMesh()
        target_mesh.vertices = o3d.utility.Vector3dVector(vertices * voxelsize)
        target_mesh.triangles = o3d.utility.Vector3iVector(faces)
        target_mesh.vertex_colors = o3d.utility.Vector3dVector(vertcolors)
        target_mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(
            os.path.join("./meshes/", "src_mesh.ply" ), target_mesh)


    save_partial_voxels = False
    if save_partial_voxels :
        c_src = 1 - (par_sdf + 3 )/6
        color =   cmap ( c_src )[:,:3]
        voxel_mesh = get_voxel_mesh(par_locs  * voxelsize, voxelsize, factor=1, color=color)
        output_ply = "./meshes/" + "partial_sdf.ply"
        o3d.io.write_triangle_mesh(output_ply, voxel_mesh)


    save_par_flow = False
    if save_par_flow :
        par_mask = np.abs(par_sdf) < 0.2
        par_locs = par_locs[par_mask]
        par_flow = dense_flow[par_locs[:, 0], par_locs[:, 1], par_locs[:, 2]]
        flow_cmap = compute_3d_norm_vector_cmap(par_flow)
        flow_mesh = get_motion_mesh(par_locs * voxelsize, par_flow,
                                    flow_cmap,
                                    radius=voxelsize * 0.12)

        output_ply = "meshes/" + level + camera+"_field.ply"
        o3d.io.write_triangle_mesh(output_ply, flow_mesh)









