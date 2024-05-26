### volumetric data with complete shape and motion
189 sequences for humanoids from mixamo (Early version of the Deformingthings4D)
```
|--091_Turn_Right
    |--extr_c2w_tum.txt    #camera extrinsics
    |--intr_c2w_tum.txt    #camera intrinsics   (42 cameras in total)
    |--tsdf
        |--00000_10_001_tsdf.bin  # partial tsdf data from a single depth image [frame_id]_[voxel_size(*0.1cm)]_[camera ID]
        |--00000_10_tsdf.bin  # complete tsdf data [frame_id]_[voxel_size(*0.1cm)]
        |--00000_10_flow.bin  # volumetric motion field [frame_id]_[voxel_size(*0.1cm)]
        |--00000_20_mc.obj   # mesh obtained via marching cube at voxel size 2cm
    |--depth
        |--00000_001.png  #raw depth image, [frame_id]_[camera ID]
```


### Load _tsdf.bin file
```python
import struct
import numpy as np
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
```

### Load _flow.bin file

```python
import struct
import numpy as np
def load_motion(file, is_npz = False):
    fin = open(file, 'rb')
    num = struct.unpack('Q', fin.read(8))[0]
    flow = struct.unpack('f'*num*3, fin.read(num*3*4))
    flow = np.asarray(flow, dtype=np.float32).reshape([num, 3])
    # flow = np.flip(flow,1).copy() # convert to zyx ordering
    fin.close()
    return flow
```