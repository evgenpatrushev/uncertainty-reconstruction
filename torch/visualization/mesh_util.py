import numpy as np
import open3d as o3d

def get_cross_prod_mat(pVec_Arr):
    # pVec_Arr shape (3)
    qCross_prod_mat = np.array([
        [0, -pVec_Arr[2], pVec_Arr[1]],
        [pVec_Arr[2], 0, -pVec_Arr[0]],
        [-pVec_Arr[1], pVec_Arr[0], 0],
    ])
    return qCross_prod_mat


def caculate_align_mat(pVec_Arr):
    scale = np.linalg.norm(pVec_Arr)
    pVec_Arr = pVec_Arr/ scale
    # must ensure pVec_Arr is also a unit vec.
    z_unit_Arr = np.array([0,0,1])
    z_mat = get_cross_prod_mat(z_unit_Arr)

    z_c_vec = np.matmul(z_mat, pVec_Arr)
    z_c_vec_mat = get_cross_prod_mat(z_c_vec)

    qTrans_Mat = np.eye(3,3) + z_c_vec_mat + np.matmul(z_c_vec_mat, z_c_vec_mat)/(1 + np.dot(z_unit_Arr, pVec_Arr))
    # qTrans_Mat *= scale
    return qTrans_Mat


def _o3d_arrow (begin, end, radius=0.2):


    vec_Arr = np.array(end) - np.array(begin)
    vec_len = np.linalg.norm(vec_Arr)


    mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(
        cone_height= vec_len*0.15, #vec_len*0.35,
        cone_radius=radius*2,
        cylinder_height= vec_len*0.85,#vec_len*0.65,
        cylinder_radius=radius,resolution=3,cylinder_split=1
    )


    mesh_arrow.paint_uniform_color([1, 0, 1])
    mesh_arrow.compute_vertex_normals()

    # mesh_sphere_begin = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=20)
    # mesh_sphere_begin.translate(begin)
    # mesh_sphere_begin.paint_uniform_color([0, 1, 1])
    # mesh_sphere_begin.compute_vertex_normals()
    #
    # mesh_sphere_end = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=20)
    # mesh_sphere_end.translate(end)
    # mesh_sphere_end.paint_uniform_color([0, 1, 1])
    # mesh_sphere_end.compute_vertex_normals()

    rot_mat = caculate_align_mat(vec_Arr)
    mesh_arrow.rotate(rot_mat, center=False)
    mesh_arrow.translate(np.array(begin))

    return  [ mesh_arrow ]

def o3d_camera_mesh ( extrinsics = None ) :

    O = [0, 0, 0]
    A = [-.1, -.1, .2]
    B = [ .1, -.1, .2]
    C = [ .1,  .1, .2]
    D = [-.1,  .1, .2]

    edge = [(O,A), (O,B), (O,C), (O,D),
            (A,B), (B,C), (C,D), (D,A)]

    edge_meshes = []

    for e in edge:
        # print (e)
        start , end = e
        mesh_cylinder = _o3d_cylinder(start, end, radius= 0.02)[0]
        edge_meshes.append(mesh_cylinder)

    edge_meshes = merge_meshes(edge_meshes)

    if extrinsics is not None :
        pass # apply transformations

    return edge_meshes




def _o3d_cylinder (begin, end, radius=0.2, color = [0.5, 0.5, 0.5]):


    vec_Arr = np.array(end) - np.array(begin)
    vec_len = np.linalg.norm(vec_Arr)


    mesh_cylinder = o3d.geometry.TriangleMesh.create_arrow(
        cone_height=vec_len*0.001,
        cone_radius=radius*0.001,
        cylinder_height= vec_len,
        cylinder_radius=radius
    )


    mesh_cylinder.paint_uniform_color(color )
    mesh_cylinder.compute_vertex_normals()

    # mesh_sphere_begin = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=20)
    # mesh_sphere_begin.translate(begin)
    # mesh_sphere_begin.paint_uniform_color([0, 1, 1])
    # mesh_sphere_begin.compute_vertex_normals()
    #
    # mesh_sphere_end = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=20)
    # mesh_sphere_end.translate(end)
    # mesh_sphere_end.paint_uniform_color([0, 1, 1])
    # mesh_sphere_end.compute_vertex_normals()

    rot_mat = caculate_align_mat(vec_Arr)
    mesh_cylinder.rotate(rot_mat, center=False)
    mesh_cylinder.translate(np.array(begin))

    return  [ mesh_cylinder ]

def get_voxel_mesh(voxel_locs, voxel_size =1, factor =0.9, transform=None, color = None):
    '''
    :param voxel_locs: [n,3]
    :param voxel_size:
    :param factor:
    :param transform:
    :param color : N,3
    :return:
    '''
    # collect verts from sdf

    verts = voxel_locs[:, :3]

    verts = np.stack(verts).astype(np.float32)
    # verts = verts[:, ::-1] + 0.5

    n_cube = verts.shape[0]

    verts = np.expand_dims( verts, axis=1 ).repeat(8,axis=1)

    offset = np.array(
                [[0, 0, 0],
                [1, 0, 0],
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 0],
                [1, 1, 0],
                [0, 1, 1],
                [1, 1, 1]])[np.newaxis,:] - 0.5
             # * 0.8 * factor
    offset = offset* voxel_size * factor

    verts = verts + offset
    verts = verts.reshape ([-1,3])

    # vert_color = None
    # if color is not None :
    #     vert_color =

    # verts = np.array([tuple(v) for v in verts], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    # verts = plyfile.PlyElement.describe(verts, 'vertex')


    faces= np.array([[4, 7, 5],
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


    faces = faces.reshape([1,-1,3]).repeat(n_cube, axis=0) #[n,11,3]
    # vert_offset = np.arange()
    vert_offset = np.arange(n_cube).reshape([-1,1,1]) * 8 #[n,1, 1]
    faces = faces + vert_offset
    faces = faces.reshape ([-1, 3])



    # faces_array = np.empty(len(faces), dtype=[('vertex_indices', 'i4', (3,))])
    # faces_array['vertex_indices'] = faces

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts )  # [N,3]
    mesh.triangles = o3d.utility.Vector3iVector(faces) #[M,3]
    # mesh.vertex_colors = o3d.utility.Vector3dVector(colors )
    if color is not None :
        # verts = np.expand_dims(verts, axis=1).repeat(8, axis=1)
        color = np.expand_dims ( color, axis= 1 ).repeat(8, axis=1)
        color = color.reshape([-1, 3])
        mesh.vertex_colors = o3d.utility.Vector3dVector(color)
    mesh.compute_vertex_normals()
    return mesh




def merge_meshes ( mesh_lists):

    verts = []
    triangles = []
    colors = []

    index_offest = 0

    for mesh in mesh_lists  :

        new_vert = np.asarray( mesh.vertices)
        new_face = np.asarray( mesh.triangles)
        new_color = np.asarray ( mesh.vertex_colors )
        new_face = new_face + index_offest

        verts.append(new_vert)
        triangles.append(new_face)
        colors.append(new_color)

        index_offest = index_offest + new_vert.shape[0]

    verts = np.concatenate ( verts, axis=0 )
    triangles = np.concatenate ( triangles, axis=0)
    colors = np.concatenate ( colors, axis=0)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts )
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors )
    # mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh])
    return mesh

def get_motion_mesh ( locs , flow, flow_cmap, radius ):
    ''' input are normalized by voxel size
    :param locs: [num, 3]
    :param flow: [num, 3]
    :param flow_cmap: [num, 3]
    :return:
    '''
    num, _ = locs.shape
    verts = []
    triangles = []
    colors = []

    print ("total number", num)
    for i in range ( num ):
        if i%1000 == 0 and i > 0 :
            print ("current id", i)
        start = locs [i]
        displacement = flow [i]
        displacement = displacement
        length = np.linalg.norm(displacement)
        # length = length/voxel_size
        end = start + displacement

        if length < radius * 8 :
            mesh_arrow = _o3d_arrow(start, end, radius= length * 0.12)[0]
        else :
            mesh_arrow = _o3d_arrow(start, end, radius= radius)[0]


        new_vert = np.asarray( mesh_arrow.vertices)
        new_face = np.asarray( mesh_arrow.triangles)

        rand_color = np.array( [flow_cmap[i]] )
        rand_color = np.repeat ( rand_color, len(new_vert), axis = 0)

        index_offset = len(verts)
        index_offset = index_offset * new_vert.shape[0]
        new_face = new_face + index_offset


        verts.append(new_vert)
        triangles.append(new_face)
        colors.append(rand_color)

    verts = np.concatenate ( verts, axis=0 )
    triangles = np.concatenate ( triangles, axis=0)
    colors = np.concatenate ( colors, axis=0)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts )
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors )
    mesh.compute_vertex_normals()

    # o3d.visualization.draw_geometries([mesh])

    return mesh


def compute_3d_norm_vector_cmap (vector):
    '''
    :param vector:  [n, 3]
    :return: [n,3]
    '''
    vector_norm = np.linalg.norm ( vector , axis=1, keepdims=True)
    vector_cmap = vector / vector_norm
    vector_cmap = (vector_cmap +1) / 2
    # vector_cmap = vector_cmap * 0.3 + 0.5
    return  vector_cmap

def flow2mesh (output_ply, pred_locs, pred_sdf, pred_mot, voxel_size, threshold = 0.5, radius_ratio = 0.12 ):
    ''' all input are voxel-metric
    :param output_ply:
    :param pred_locs: [N, 3]
    :param pred_sdf: [N,]
    :param pred_motion: [N, 3]
    :param voxel_size:
    :param threshold:
    :return:
    '''

    tgt_mask = pred_sdf  < threshold
    tgt_locs = pred_locs [tgt_mask]
    tgt_flow = pred_mot[tgt_mask]
    tgt_flow_cmap = compute_3d_norm_vector_cmap(tgt_flow)
    flow_mesh = get_motion_mesh(tgt_locs , tgt_flow,
                                tgt_flow_cmap,
                                radius=  radius_ratio)
    # output_ply = os.path.join(args.output, "_pred_field.ply" )
    o3d.io.write_triangle_mesh(output_ply, flow_mesh)

def depth2pcl(output_ply, pc, color=None):
    '''
    :param output_ply:
    :param pc:[N, 3]
    :param color:
    :return:
    '''

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector( pc )
    if color is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(color)
    mesh.compute_vertex_normals()



def get_bBox_mesh ( target_locs, radius=0.1, padding = None ):
    '''
    :param target_locs: [n, 3]
    :param radius:
    :return:
    '''

    x_min, x_max = target_locs[:, 0].min(), target_locs[:, 0].max()
    y_min, y_max = target_locs[:, 1].min(), target_locs[:, 1].max()
    z_min, z_max = target_locs[:, 2].min(), target_locs[:, 2].max()

    if padding is not None :
        x_min = x_min - padding
        y_min = y_min - padding
        z_min = z_min - padding

        x_max = x_max + padding
        y_max = y_max + padding
        z_max = z_max + padding


    A = [x_min, y_min, z_min]
    B = [x_min, y_min, z_max]
    C = [x_max, y_min, z_min]
    D = [x_max, y_min, z_max]

    E = [x_min, y_max, z_min]
    F = [x_min, y_max, z_max]
    G = [x_max, y_max, z_min]
    H = [x_max, y_max, z_max]

    edge = [ (A,B), (A,C), (A,E), (B,D),
             (B,F), (C,D), (C,G), (D,H),
             (E,F), (E,G), (F,H), (G,H)]

    edge_meshes = []

    for e in edge:
        start , end = e
        mesh_cylinder = _o3d_cylinder(start, end, radius= radius)[0]
        edge_meshes.append(mesh_cylinder)

    edge_meshes = merge_meshes(edge_meshes)
    return edge_meshes


def get_color_sphere():
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=30)
    verts = np.asarray ( sphere.vertices )
    colors = compute_3d_norm_vector_cmap(verts)
    sphere.vertex_colors = o3d.utility.Vector3dVector(colors)
    # sphere.compute_vertex_normals()
    # o3d.visualization.draw_geometries([sphere])
    # o3d.io.write_triangle_mesh("color_sphere.ply", sphere)
    return sphere

def node_o3d_spheres (node_array, r=0.1, resolution=10, color = [1, 0. , 0.]):
    '''
    :param node_array: [n,3]
    :param r:
    :param resolution:
    :param color:
    :return:
    '''

    N, _ = node_array.shape

    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=resolution)

    vertices  = np.asarray(mesh_sphere.vertices)   # point 3d
    triangles = np.asarray(mesh_sphere.triangles) # index

    num_sphere_vertex, _ = vertices.shape

    vertices = np.expand_dims (vertices, axis=0)
    triangles = np.expand_dims (triangles, axis=0)

    vertices = np.repeat ( vertices , [N], axis=0) # change corr 3D
    triangles = np.repeat ( triangles ,[N], axis=0) # change index

    # reposition vertices
    node_array = np.expand_dims( node_array , axis=1)
    vertices = node_array + vertices
    vertices = vertices.reshape( [-1, 3])

    # change index
    index_offset = np.arange(N).reshape( N, 1, 1) * num_sphere_vertex
    triangles = triangles + index_offset
    triangles = triangles.reshape( [-1, 3])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()

    mesh.paint_uniform_color(color)

    # # o3d.visualization.draw_geometries([ mesh ])
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(mesh)
    # vis.get_render_option().load_from_json("./renderoption.json")
    # vis.run()
    # # vis.capture_screen_image("output/ours_silver-20.jpg")
    # vis.destroy_window()
    return mesh