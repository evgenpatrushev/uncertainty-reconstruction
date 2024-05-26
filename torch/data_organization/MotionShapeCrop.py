import os, sys, struct
import numpy as np
import torch
import random
import math

import  sys, os
sys.path.append("../")
from util.geometry import read_tum_pose, quat2mat, depth_2_pc

from visualization.mesh_util import get_voxel_mesh, get_motion_mesh, compute_3d_norm_vector_cmap,get_bBox_mesh
import open3d as o3d



n_hierachy = 3

chunk_dimz, chunk_dimy, chunk_dimx = 128, 96, 96  #-> 64,48,48 -> 32, 24, 24 -> 16, 12, 12
sample_interval = int(chunk_dimx/(2**(n_hierachy-1)))  # sample every 8 voxels of size 8cm
sample_interval_z = int(chunk_dimz/8)  # sample every 8 voxels of size 8cm


voxel_size = 20  # 05 10 20 40 80
# cube_sizes = [0.01, 0.02, 0.04, 0.08]
cube_sizes = [ 0.02, 0.04, 0.08 ]
dimension = 128
DEPTH_SCALE = 1000.


NUM_CAMERA = 42

from matplotlib import cm

cmap = cm.get_cmap('jet')

# viewpoints = [24, 31, 22, 29]
viewpoints = [24, 31, 22, 29]
# viewpoints = np.arange(1,36).tolist()
# viewpoints = np.arange(1,NUM_CAMERA).tolist()


# viewpoints = [24, 31, 26, 29, 22, 23]

def load_viewpoint_sphere(folder = "./"):

    '''load camera extrinsics'''
    trajectory = os.path.join( folder, "extr_c2w_tum.txt")
    cameras = read_tum_pose(trajectory)
    cam_extrinsics = []
    for camera_id in range(NUM_CAMERA):
        _, cam = cameras [camera_id]
        trn = np.array( [float(cam[0]), float (cam[1]), float(cam[2])]).reshape([3,1])
        qx, qy, qz, qw = float(cam[3]), float(cam[4]), float(cam[5]), float(cam[6])
        rot = quat2mat([qw, qx, qy, qz])
        extrinsics = np.concatenate ( [rot, trn], axis= 1)
        extrinsics = np.concatenate ( [ extrinsics, np.array([[0,0,0,1]])] , axis=0)
        cam_extrinsics.append(extrinsics)
    return cam_extrinsics

CAM_INTRIN_AZURE = [443.405000, 256.000000, 443.405000, 256.000000]


# locs: xyz ordering
def sparse_to_dense_np_xyz(locs, values, dimx, dimy, dimz, default_val):
    nf_values = 1 if len(values.shape) == 1 else values.shape[1]
    dense = np.zeros([dimx, dimy, dimz, nf_values], dtype=values.dtype)
    dense.fill(default_val)
    #print('dense', dense.shape)
    #print('locs', locs.shape)
    #print('values', values.shape)
    dense[locs[:,0], locs[:,1], locs[:,2],:] = values
    if nf_values > 1:
        return dense.reshape([dimx, dimy, dimz, nf_values])
    return dense.reshape([dimx, dimy, dimz])

def compute_IOU ( src, tgt, thresh = 100):
    src_mask = (np.abs(src) < thresh).float()
    tgt_mask = (np.abs(tgt) < thresh).float()
    return src.sum() / tgt_mask.sum()

def dense_to_sparse_np(grid, thresh):
    locs = np.where(np.abs(grid) < thresh)
    values = grid[locs[0], locs[1], locs[2]]
    locs = np.stack([locs[0], locs[1], locs[2]] )
    return locs, values

def dense_to_sparse_np_motion(grid, thresh):
    '''grid : [dx,dy,dz,3]'''
    locs = np.where(np.abs(grid[:,:,:,0]) < thresh)
    values = grid[locs[0], locs[1], locs[2]]
    locs = np.stack([locs[0], locs[1], locs[2]] )
    return locs, values

def load_scene(file):
    fin = open(file, 'rb')
    dimx = struct.unpack('Q', fin.read(8))[0]
    dimy = struct.unpack('Q', fin.read(8))[0]
    dimz = struct.unpack('Q', fin.read(8))[0]
    voxelsize = struct.unpack('f', fin.read(4))[0] # in meters
    world2grid = struct.unpack('f'*4*4, fin.read(4*4*4))
    world2grid = np.asarray(world2grid, dtype=np.float32).reshape([4, 4])
    # data
    num = struct.unpack('Q', fin.read(8))[0]
    locs = struct.unpack('I'*num*3, fin.read(num*3*4))
    locs = np.asarray(locs, dtype=np.int32).reshape([num, 3])
    # locs = np.flip(locs,1).copy() # convert to zyx ordering
    sdf = struct.unpack('f'*num, fin.read(num*4))
    sdf = np.asarray(sdf, dtype=np.float32)
    fin.close()
    return [locs, sdf], [dimz, dimy, dimx], world2grid, voxelsize



def load_motion(file):

    fin = open(file, 'rb')
    num = struct.unpack('Q', fin.read(8))[0]
    flow = struct.unpack('f'*num*3, fin.read(num*3*4))
    flow = np.asarray(flow, dtype=np.float32).reshape([num, 3])
    # flow = np.flip(flow,1).copy() # convert to zyx ordering
    fin.close()
    return flow


def compute_chunk_offsets(min_index, max_index, chunk_dim, sample_interval, drop_thresh = 0.15 ):


    len_ = max_index - min_index + 1
    float_pos = (len_ - chunk_dim) / sample_interval + 1
    num_offsets_z = math.ceil(float_pos)
    need_padding = (num_offsets_z != math.floor(float_pos))
    residual_rate = float_pos - math.floor(float_pos)
    offsets = np.arange(num_offsets_z) * sample_interval
    if num_offsets_z > 0 and need_padding:
        offsets[-1] = len_ - chunk_dim

    if need_padding and residual_rate< drop_thresh : #hold_tail==False :
        offsets = offsets [:-1]

    return offsets.tolist()






def processSequence ( seq_path, dump_path, save_mesh=False ):

    CAM_EXTRINSICS = load_viewpoint_sphere()

    seq_name = seq_path.split("/")[-1]

    print ("processing...", seq_name)

    scan_root = os.path.join(seq_path, "tsdf")


    for frame_id  in range (5000) :


        target_sdfs = []
        target_locs = []
        target_locs_wld = []
        dense_target_sdf = []
        grid2world_matrices = None
        target_motion_wld = []
        dense_target_motion_wld = []

        frame_id = "%05d" % frame_id


        check_sdf = os.path.join(scan_root, '_'.join( [ frame_id, "10_flow.bin" ]) )
        if not os.path.exists(check_sdf):
            print(seq_name, "finished")
            break

        """Load target SDFs / motion for 1 levels"""
        for level in range(n_hierachy):

            level_size = "%02d" % (voxel_size * 2 ** level)

            tgt_sdf = os.path.join(scan_root, '_'.join( [ frame_id, level_size, "tsdf.bin" ]) )
            inputs, dims, world2grid, _ = load_scene(tgt_sdf)
            tgt_locs, tgt_sdf = inputs
            current_dim = int ( dimension / (2**level) )
            dense_target_i = sparse_to_dense_np_xyz(tgt_locs, tgt_sdf.reshape([-1, 1]), current_dim, current_dim, current_dim, -float('inf'))
            grid2world = np.linalg.inv(world2grid)

            tgt_motion = os.path.join(scan_root, frame_id + "_" + level_size + "_flow.bin")
            tgt_motion = load_motion(tgt_motion)

            """convert to world corrdinate"""
            # tgt_locs_wld = tgt_locs * cube_sizes[level]
            # ones = np.ones([tgt_locs_wld.shape[0], 1])
            # tgt_locs_wld = np.concatenate([tgt_locs_wld, ones], axis=1)
            # tgt_locs_wld = np.matmul(grid2world, tgt_locs_wld.transpose())
            tgt_motion_wld = np.matmul(grid2world[:3, :3], tgt_motion.transpose())

            dense_tgt_motion_wld = sparse_to_dense_np_xyz \
                (tgt_locs, tgt_motion_wld.transpose(), current_dim, current_dim, current_dim, -float('inf'))

            target_locs.append(tgt_locs)
            target_sdfs.append(tgt_sdf)
            dense_target_sdf.append(dense_target_i)
            grid2world_matrices = grid2world


            # target_locs_wld.append(tgt_locs_wld)
            target_motion_wld.append(tgt_motion_wld)
            dense_target_motion_wld.append(dense_tgt_motion_wld)



        '''get target bounding box'''
        x_min, x_max = target_locs[-1][:, 0].min(), target_locs[-1][:, 0].max()
        y_min, y_max = target_locs[-1][:, 1].min(), target_locs[-1][:, 1].max()
        z_min = 0

        """sample chunks [16,8,8]"""
        """skip every ? voxels,  z axis: height; x axis: width; y axis: depth; """
        # offset_z = compute_chunk_offsets(z_min, z_max, chunk_dim=int(chunk_dimz/(2**(n_hierachy-1))), sample_interval=sample_interval_z )
        offset_z = [0]
        offset_y = compute_chunk_offsets(y_min, y_max, chunk_dim=int(chunk_dimy/(2**(n_hierachy-1))), sample_interval=sample_interval )
        offset_x = compute_chunk_offsets(x_min, x_max, chunk_dim=int(chunk_dimx/(2**(n_hierachy-1))), sample_interval=sample_interval )


        for view_id in viewpoints :

            source_locs = []
            source_sdfs = []
            dense_source_sdf = []
            dense_source_motion_wld = []
            cam_extrinsics = CAM_EXTRINSICS [view_id]

            view_id = "%03d" % view_id

            """get one level sdfs/flow for source"""
            for level in  range(1):

                level_size =  "%02d"  % (voxel_size * 2**level)
                current_dim = int(dimension / (2 ** level))


                src_sdf = os.path.join( scan_root,  '_'.join( [ frame_id, level_size, view_id, "tsdf.bin" ]) )
                inputs, dims, world2grid, _= load_scene(src_sdf)
                src_locs , src_sdf =  inputs
                dense_source_i = sparse_to_dense_np_xyz(src_locs, src_sdf.reshape([-1, 1]), current_dim, current_dim, current_dim, -float('inf'))


                '''`load` motion'''
                current_dim = int(dimension / (2 ** level))
                flatlocs =  src_locs[:, 0] * current_dim * current_dim + src_locs[:, 1] * current_dim + src_locs[:, 2]
                src_motion = dense_target_motion_wld[0].reshape(-1, 3)[flatlocs]
                # to deal with src voxels that is not in tgt locs
                src_motion[ np.where(np.isinf(src_motion[:,0]))] = 0
                dense_src_motion_wld = sparse_to_dense_np_xyz \
                    (src_locs, src_motion, current_dim, current_dim, current_dim, -float('inf'))


                dense_source_sdf.append(dense_source_i)
                source_locs.append(src_locs)
                source_sdfs.append(src_sdf)
                dense_source_motion_wld.append(dense_src_motion_wld)



            for off_z in offset_z :
                for off_y in offset_y :
                    for off_x in offset_x :


                        # print ("chunk id", chunk_ID)

                        z_begin = z_min + off_z
                        y_begin = y_min + off_y
                        x_begin = x_min + off_x

                        if x_begin < 0 :
                            x_begin = 0
                        if y_begin < 0 :
                            y_begin = 0

                        h_scale = 2**(n_hierachy-1)

                        tgt_sdf_chunk_dense = dense_target_sdf[ 0 ] [
                                      x_begin*h_scale : x_begin*h_scale + chunk_dimx,
                                      y_begin*h_scale : y_begin*h_scale + chunk_dimy,
                                      z_begin*h_scale : z_begin*h_scale + chunk_dimz ]

                        tgt_motion_chunk_dense = dense_target_motion_wld[ 0 ] [
                                      x_begin*h_scale : x_begin*h_scale + chunk_dimx,
                                      y_begin*h_scale : y_begin*h_scale + chunk_dimy,
                                      z_begin*h_scale : z_begin*h_scale + chunk_dimz, : ]


                        src_sdf_chunk_dense = dense_source_sdf[ 0 ] [
                                      x_begin*h_scale : x_begin*h_scale + chunk_dimx,
                                      y_begin*h_scale : y_begin*h_scale + chunk_dimy,
                                      z_begin*h_scale : z_begin*h_scale + chunk_dimz ]

                        src_motion_chunk_dense = dense_source_motion_wld[ 0 ] [
                                      x_begin*h_scale : x_begin*h_scale + chunk_dimx,
                                      y_begin*h_scale : y_begin*h_scale + chunk_dimy,
                                      z_begin*h_scale : z_begin*h_scale + chunk_dimz, : ]






                        tgt_chunk_locs, tgt_chunk_sdf = dense_to_sparse_np( tgt_sdf_chunk_dense, 100) # 100 meter
                        src_chunk_locs, src_chunk_sdf = dense_to_sparse_np( src_sdf_chunk_dense, 100)






                        tgt_chunk_locs, tgt_motion_chunk_sparse = dense_to_sparse_np_motion (tgt_motion_chunk_dense, 100 ) # 100 meters
                        src_chunk_locs, src_motion_chunk_sparse = dense_to_sparse_np_motion (src_motion_chunk_dense, 100 )




                        chunk_ID = "_".join([seq_name, frame_id, view_id, str(off_z), str(off_y), str(off_x)])


                        '''get three hirachy levels'''
                        sdf_hierarchy = []
                        mot_hierarchy = []
                        for i in range (1,n_hierachy) :

                            h_scale = 2 ** (n_hierachy - 1 - i)

                            chunk_sdf_level_i = dense_target_sdf[i][
                                                x_begin * h_scale: x_begin * h_scale + int (chunk_dimx/(2**i)),
                                                y_begin * h_scale: y_begin * h_scale + int (chunk_dimy/(2**i)),
                                                z_begin * h_scale: z_begin * h_scale + int (chunk_dimz/(2**i))]

                            chunk_motion_level_i = dense_target_motion_wld[i] [
                                                x_begin * h_scale: x_begin * h_scale + int(chunk_dimx / (2 ** i)),
                                                y_begin * h_scale: y_begin * h_scale + int(chunk_dimy / (2 ** i)),
                                                z_begin * h_scale: z_begin * h_scale + int(chunk_dimz / (2 ** i)), :]

                            sdf_hierarchy.append( chunk_sdf_level_i )
                            mot_hierarchy.append( chunk_motion_level_i)

                            # save_mesh = True
                            if save_mesh:

                                chunk_locs, _ = dense_to_sparse_np(chunk_sdf_level_i, 1)
                                # data_util.visualize_sparse_sdf_as_cubes(chunk_locs.transpose() * (2**i),"./meshes/" + chunk_ID + "_level_" + str(i) + "_sdf.ply", factor = 2**i)
                                ply_name = "./meshes/" + chunk_ID + "_level_" + str(i) + "_sdf.ply"

                                voxel_mesh = get_voxel_mesh(chunk_locs.transpose()*cube_sizes[i], cube_sizes[i])
                                o3d.io.write_triangle_mesh(ply_name, voxel_mesh)

                                _, chunk_motion = dense_to_sparse_np_motion(chunk_motion_level_i, 100)  # 100 meters
                                tgt_flow_cmap = compute_3d_norm_vector_cmap(chunk_motion)
                                flow_mesh = get_motion_mesh(chunk_locs.transpose() * cube_sizes[i], chunk_motion,
                                                            tgt_flow_cmap,
                                                            radius=cube_sizes[i] * 0.12)
                                output_ply = "meshes/" + chunk_ID + "_level_" + str(i) + "_tgt_field.ply"
                                o3d.io.write_triangle_mesh(output_ply, flow_mesh)


                        """save voxels mesh to files"""
                        if save_mesh:
                            print ("chunk_ID", chunk_ID)
                            tgt_ply_name = "./meshes/" + chunk_ID + "_tgt_sdf.ply"
                            src_ply_name = "./meshes/" + chunk_ID + "_src_sdf.ply"
                            bbox_ply_name = "./meshes/" + chunk_ID + "_tgt_box.ply"
                            c_src = 1 - (src_chunk_sdf/cube_sizes[0] + 3) / 6
                            color = cmap(c_src)[:, :3]
                            voxel_mesh = get_voxel_mesh(src_chunk_locs.transpose()*cube_sizes[0], cube_sizes[0],factor=0.7, color=color)
                            o3d.io.write_triangle_mesh(src_ply_name, voxel_mesh)
                            c_src = 1 - (tgt_chunk_sdf / cube_sizes[0] + 3) / 6
                            color = cmap(c_src)[:, :3]
                            voxel_mesh = get_voxel_mesh(tgt_chunk_locs.transpose()*cube_sizes[0], cube_sizes[0],factor=0.7, color=color)
                            o3d.io.write_triangle_mesh(tgt_ply_name, voxel_mesh)
                            tgt_bBox = get_bBox_mesh(tgt_chunk_locs.transpose()*cube_sizes[0],radius=0.005)
                            o3d.io.write_triangle_mesh(bbox_ply_name, tgt_bBox)


                            """render tgt motion to ply"""
                            tgt_mask = np.abs(tgt_chunk_sdf / cube_sizes[0]) < 0.2
                            tgt_locs = tgt_chunk_locs.transpose()[tgt_mask]
                            tgt_flow = tgt_motion_chunk_sparse[tgt_mask]
                            tgt_flow_cmap = compute_3d_norm_vector_cmap(tgt_flow)
                            flow_mesh = get_motion_mesh(tgt_locs * cube_sizes[0], tgt_flow,
                                                        tgt_flow_cmap,
                                                        radius=cube_sizes[0] * 0.12)
                            output_ply = "meshes/" + chunk_ID + "_tgt_field.ply"
                            o3d.io.write_triangle_mesh(output_ply, flow_mesh)


                            """render src motion to ply"""
                            src_mask = np.abs(src_chunk_sdf / cube_sizes[0]) < 0.1
                            src_locs = src_chunk_locs.transpose()[src_mask]
                            src_flow = src_motion_chunk_sparse[src_mask]
                            src_flow_cmap = compute_3d_norm_vector_cmap(src_flow)
                            flow_mesh = get_motion_mesh(src_locs * cube_sizes[0], src_flow,
                                                        src_flow_cmap,
                                                        radius=cube_sizes[1] * 0.12)
                            output_ply = "meshes/" + chunk_ID + "_src_field.ply"
                            o3d.io.write_triangle_mesh(output_ply, flow_mesh)

                        # exit(0)


                        '''save training data'''
                        crop_x, crop_y,crop_z = tgt_sdf_chunk_dense.shape
                        if crop_z == chunk_dimz and crop_y == chunk_dimy and crop_x == chunk_dimx :
                            fname = os.path.join(dump_path, chunk_ID + ".npz", )
                            np.savez_compressed(fname,
                                                dims=np.array([chunk_dimx, chunk_dimy, chunk_dimz]),
                                                input_locs=src_chunk_locs,
                                                input_motion= src_motion_chunk_sparse,
                                                target_locs=tgt_chunk_locs,
                                                tgt_motion = tgt_motion_chunk_sparse,
                                                input_sdf=src_chunk_sdf,
                                                tgt_sdf_dense = tgt_sdf_chunk_dense,
                                                h0_sdf_dense=sdf_hierarchy[0],
                                                h1_sdf_dense=sdf_hierarchy[1],
                                                h0_mot_dense=mot_hierarchy[0],
                                                h1_mot_dense=mot_hierarchy[1])





if __name__ == '__main__':
    import os,glob,sys
    import multiprocessing

    pool = multiprocessing.Pool(processes=20)
    dataset_path = "../../data/DeformingThings4D"
    chunk_prefix = "MotionShape_2cm"

    raw_sequence_folder = os.path.join( dataset_path, "raw")
    dumping_folder = os.path.join( dataset_path, "chunks" , "_".join( [chunk_prefix, str(chunk_dimx), str(chunk_dimy), str(chunk_dimz)]) )

    if not os.path.exists(dumping_folder):
        os.makedirs(dumping_folder)

    raw_sequence = glob.glob( raw_sequence_folder + "/*", recursive=False)

    for seq in raw_sequence :

        pool.apply_async(processSequence, args=(seq, dumping_folder))

    pool.close()
    pool.join()