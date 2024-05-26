
import  sys, os
sys.path.append("../")
import data_util

if __name__ == '__main__':

    max_height = 100
    up_axis = 0

    sdf_file = "/home/adl4cv/003_Capoeira/tsdf/00000_20_028_tsdf.bin"
    output_ply = "/home/adl4cv/projects/sdfs28.ply"
    inputs, dims, world2grid = data_util.load_scene(sdf_file)
    locs , sdf =  inputs
    dimz, dimy, dimx = dims

    if dimz > max_height :
        mask_input = locs[:, up_axis] < max_height
        mask_input = mask_input * (locs[:, 1] < max_height )
        mask_input = mask_input * (locs[:, 2] < max_height)

        locs  = locs[mask_input]
        sdf  = sdf[mask_input]


    for i in range (-4,4):
        level_ply = "level"+ str(i+1) + ".ply"

        mask = sdf > i+0.5
        mask = mask * (sdf < i+1.5)

        data_util.visualize_sparse_sdf_as_cubes ( locs [mask],  level_ply )

    data_util.visualize_sparse_sdf_as_cubes ( locs, output_ply )

    # data_util.visualize_sparse_sdf_as_points ( locs, sdf, 2, output_ply )