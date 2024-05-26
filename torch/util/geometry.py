import numpy as np


def depth_2_pc (depth , intrin ):

    fx, cx, fy, cy = intrin
    height , width = depth.shape
    u = np.arange (width) * np.ones(  [height, width ])
    v = np.arange (height) * np.ones( [width, height])
    v = np.transpose( v )

    X = (u - cx ) * depth / fx
    Y = (v - cy ) * depth / fy
    Z = depth
    return  np.stack ( [X, Y, Z], -1)


def quat2mat(q):
    w, x, y, z = q
    Nq = w * w + x * x + y * y + z * z
    if Nq < 1e-8:
        return np.eye(3)
    s = 2.0 / Nq
    X = x * s
    Y = y * s
    Z = z * s
    wX = w * X;
    wY = w * Y;
    wZ = w * Z
    xX = x * X;
    xY = x * Y;
    xZ = x * Z
    yY = y * Y;
    yZ = y * Z;
    zZ = z * Z
    return np.array(
        [[1.0 - (yY + zZ), xY - wZ, xZ + wY],
         [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
         [xZ - wY, yZ + wX, 1.0 - (xX + yY)]])


def pc2uv (pcloud, intrinsics) :

    fx, cx, fy, cy = intrinsics

    X, Y, depth, _ = pcloud

    u = fx * X / depth + cx
    v = fy * Y / depth + cy

    return u.astype(int) , v.astype(int)

def read_tum_pose(filename):
    """
    Reads a trajectory from a text file.
    File format:
    The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
    and "d1 d2 d3.." is arbitary data (e.g., a 3D position and 3D orientation) associated to this timestamp.
    Input:
    filename -- File name
    Output:
    dict -- dictionary of (stamp,data) tuples
    """
    file = open(filename)
    data = file.read()
    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    list = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if
            len(line) > 0 and line[0] != "#"]
    list = [(float(l[0]), l[1:]) for l in list if len(l) > 1]
    return list
