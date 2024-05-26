import os, sys, struct
import scipy.io as sio
import numpy as np
import torch
import torch.utils.data
import random
import math
import plyfile
from skimage.measure import block_reduce
import data_util
from scipy.stats import special_ortho_group


def collate(batch):
    names = [x['name'] for x in batch]
    # collect sparse inputs
    locs = batch[0]['input'][0]
    locs = torch.cat([locs, torch.zeros(locs.shape[0], 1).long()], 1)
    feats = batch[0]['input'][1]
    known = None
    if batch[0]['known'] is not None:
        known = torch.stack([x['known'] for x in batch])

    colors = None

    sdf_hierarchy = None
    if batch[0]['sdf_hierarchy'] is not None:
        sdf_hierarchy = [None] * len(batch[0]['sdf_hierarchy'])
        for h in range(len(batch[0]['sdf_hierarchy'])):
            sdf_hierarchy[h] = torch.stack([x['sdf_hierarchy'][h] for x in batch])

    mot_hierarchy = None
    if batch[0]['mot_hierarchy'] is not None:
        mot_hierarchy = [None] * len(batch[0]['mot_hierarchy'])
        for h in range(len(batch[0]['mot_hierarchy'])):
            mot_hierarchy[h] = torch.stack([x['mot_hierarchy'][h] for x in batch])

    for b in range(1, len(batch)):
        cur_locs = batch[b]['input'][0]
        cur_locs = torch.cat([cur_locs, torch.ones(cur_locs.shape[0], 1).long() * b], 1)
        locs = torch.cat([locs, cur_locs])
        feats = torch.cat([feats, batch[b]['input'][1]])

    sdfs = torch.stack([x['sdf'] for x in batch])
    motion = torch.stack([x['motion'] for x in batch])

    world2grids = torch.stack([x['world2grid'] for x in batch])

    orig_dims = torch.stack([x['orig_dims'] for x in batch])

    return {'name': names,
            'input': [locs, feats], 'sdf': sdfs, 'motion': motion,
            'world2grid': world2grids, 'known': known,
            'sdf_hierarchy': sdf_hierarchy, 'mot_hierarchy': mot_hierarchy,
            'orig_dims': orig_dims}


class MotionCompleteDataset(torch.utils.data.Dataset):

    def __init__(self, files, input_dim, truncation, num_hierarchy_levels, max_input_height, num_overfit=0,
                 target_path=''):
        assert (num_hierarchy_levels <= 4)  # havent' precomputed more than this
        self.is_chunks = target_path == ''  # have target path -> full scene data
        if not target_path:
            self.files = [f for f in files if os.path.isfile(f)]
        else:
            self.files = [(f, os.path.join(target_path, os.path.basename(f))) for f in files if
                          (os.path.isfile(f) and os.path.isfile(os.path.join(target_path, os.path.basename(f))))]
        self.input_dim = input_dim
        self.truncation = truncation
        self.num_hierarchy_levels = num_hierarchy_levels
        self.max_input_height = max_input_height
        self.UP_AXIS = 0
        if num_overfit > 0:
            num_repeat = max(1, num_overfit // len(self.files))
            self.files = self.files * num_repeat

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        name = None
        if self.is_chunks:
            name = os.path.splitext(os.path.basename(file))[0]

            inputs, targets, dims, _, _, _, _ = data_util.load_train_motion_shape_complete(file)

            input_locs, input_motion = inputs
            target_locs, target_motion = targets
            dimx, dimy, dimz = dims

            self.data_augmentation = True
            if self.data_augmentation:
                rand_rot = special_ortho_group.rvs(3)
                input_motion = np.matmul(rand_rot, input_motion.T).T
                target_motion = np.matmul(rand_rot, target_motion.T).T

        else:
            raise NotImplementedError()

        '''empty padding target locations'''
        input_dense = data_util.sparse_to_dense_np_xyz(
            input_locs, input_motion, dimx, dimy, dimz, -float('inf'))
        input_mask = ~np.isinf(input_dense[:, :, :, 0])

        target_dense = data_util.sparse_to_dense_np_xyz(
            target_locs, target_motion, dimx, dimy, dimz, -float('inf'))
        target_mask = ~np.isinf(target_dense[:, :, :, 0])

        known_dense = input_mask * target_mask
        known_sparse = known_dense[target_locs[:, 0], target_locs[:, 1], target_locs[:, 2]]

        input_motion = target_motion.copy()
        input_motion[~known_sparse] = 0.0

        inputs = [torch.from_numpy(target_locs).long(), torch.from_numpy(input_motion).float()]
        target_motion = torch.from_numpy(target_motion).float()
        known_sparse = torch.from_numpy(known_sparse).bool()

        '''compute target locations in hierachy'''
        hier_mask = target_mask
        target_locs_hier = [None] * (self.num_hierarchy_levels - 1)
        for h in range(self.num_hierarchy_levels - 2, -1, -1):
            hier_mask = block_reduce(hier_mask, block_size=(2, 2, 2), func=np.max)
            hier_locs = np.where(hier_mask)
            target_locs_hier[h] = torch.from_numpy(np.stack(hier_locs, -1)).long()

        sample = {'name': name,
                  'input': inputs, 'motion': target_motion,
                  'target_locs_hier': target_locs_hier,
                  'known': known_sparse}
        return sample


def collate_mocomplete(batch):
    names = [x['name'] for x in batch]
    # collect sparse inputs
    locs = batch[0]['input'][0]
    locs = torch.cat([locs, torch.zeros(locs.shape[0], 1).long()], 1)
    feats = batch[0]['input'][1]

    known = torch.cat([x['known'] for x in batch])

    target_locs_hier = batch[0]['target_locs_hier']
    for h in range(len(target_locs_hier)):
        hier_locs = target_locs_hier[h]
        hier_locs = torch.cat([hier_locs, torch.zeros(hier_locs.shape[0], 1).long()], 1)
        for b in range(1, len(batch)):
            cur_locs = batch[b]['target_locs_hier'][h]
            cur_locs = torch.cat([cur_locs, torch.ones(cur_locs.shape[0], 1).long() * b], 1)
            hier_locs = torch.cat([hier_locs, cur_locs])
        target_locs_hier[h] = hier_locs

    for b in range(1, len(batch)):
        cur_locs = batch[b]['input'][0]
        cur_locs = torch.cat([cur_locs, torch.ones(cur_locs.shape[0], 1).long() * b], 1)
        locs = torch.cat([locs, cur_locs])
        feats = torch.cat([feats, batch[b]['input'][1]])

    motion = torch.cat([x['motion'] for x in batch])

    return {'name': names,
            'input': [locs, feats], 'motion': motion,
            'known': known,  # '6nn': _6_nei,
            'target_locs_hier': target_locs_hier,
            'bsize': len(batch)
            }


class MotionShapeDataset(torch.utils.data.Dataset):

    def __init__(self, files, input_dim, truncation, num_hierarchy_levels, max_input_height, num_overfit=0,
                 target_path=''):
        assert (num_hierarchy_levels <= 4)  # havent' precomputed more than this
        self.is_chunks = target_path == ''  # have target path -> full scene data
        if not target_path:
            self.files = [f for f in files if os.path.isfile(f)]
        else:
            self.files = [(f, os.path.join(target_path, os.path.basename(f))) for f in files if
                          (os.path.isfile(f) and os.path.isfile(os.path.join(target_path, os.path.basename(f))))]
        self.input_dim = input_dim
        self.truncation = truncation
        self.num_hierarchy_levels = num_hierarchy_levels
        self.max_input_height = max_input_height
        self.UP_AXIS = 0
        if num_overfit > 0:
            num_repeat = max(1, num_overfit // len(self.files))
            self.files = self.files * num_repeat

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        name = None
        if self.is_chunks:
            name = os.path.splitext(os.path.basename(file))[0]
            inputs, targets, dims, \
            input_sdfs, target_sdfs, sdf_hierarchy, mot_hierarchy = \
                data_util.load_train_motion_shape_complete(file)
        else:
            raise NotImplementedError()

        input_locs, input_motion = inputs
        target_locs, target_motion = targets
        dimx, dimy, dimz = dims
        input_index = np.arange(input_locs.shape[0])[:, np.newaxis]

        '''compute known mask'''
        input_dense = data_util.sparse_to_dense_np_xyz(
            input_locs, input_motion, dimx, dimy, dimz, -float('inf'))
        index_dense = data_util.sparse_to_dense_np_xyz(
            input_locs, input_index, dimx, dimy, dimz, -1)
        input_mask = ~np.isinf(input_dense[:, :, :, 0])
        target_dense = data_util.sparse_to_dense_np_xyz(
            target_locs, target_motion, dimx, dimy, dimz, -float('inf'))
        target_mask = ~np.isinf(target_dense[:, :, :, 0])
        known_dense = input_mask * target_mask
        known_sparse = known_dense[target_locs[:, 0], target_locs[:, 1], target_locs[:, 2]]
        index_sparse = index_dense[known_dense]

        inputs = [torch.from_numpy(input_locs[index_sparse]).long(),
                  torch.from_numpy(input_motion[index_sparse]).float(),
                  torch.from_numpy(input_sdfs[:, np.newaxis][index_sparse]).float()]

        target_motion = torch.from_numpy(target_motion).float()

        known_sparse = torch.from_numpy(known_sparse).bool()

        targets_sdfs = target_sdfs[np.newaxis, :]
        targets_sdfs = torch.from_numpy(targets_sdfs)
        if sdf_hierarchy is not None:
            for h in range(len(sdf_hierarchy)):
                sdf_hierarchy[h] = torch.from_numpy(sdf_hierarchy[h][np.newaxis, :])

        if mot_hierarchy is not None:
            for h in range(len(mot_hierarchy)):
                mot_hierarchy[h] = torch.from_numpy(mot_hierarchy[h][np.newaxis, :]).float()

        '''compute target locations in hierachy'''
        hier_mask = target_mask
        target_locs_hier = [None] * (self.num_hierarchy_levels - 1)
        for h in range(self.num_hierarchy_levels - 2, -1, -1):
            # maxpool3d in numpy
            hier_mask = block_reduce(hier_mask, block_size=(2, 2, 2), func=np.max)
            hier_locs = np.where(hier_mask)
            target_locs_hier[h] = torch.from_numpy(np.stack(hier_locs, -1)).long()
        target_locs_hier.append(torch.from_numpy(target_locs).long())

        sample = {'name': name,
                  'input': inputs, 'motion': target_motion,
                  'target_locs_hier': target_locs_hier,

                  'sdf': targets_sdfs, 'sdf_hierarchy': sdf_hierarchy, 'mot_hierarchy': mot_hierarchy,

                  'known': known_sparse}

        return sample


def collate_motionshape(batch):
    names = [x['name'] for x in batch]
    # collect sparse inputs
    locs = batch[0]['input'][0]
    locs = torch.cat([locs, torch.zeros(locs.shape[0], 1).long()], 1)
    feats_1 = batch[0]['input'][1]
    feats_2 = batch[0]['input'][2]

    known = torch.cat([x['known'] for x in batch])

    '''updating knn index'''
    # index_offset = 0
    # _6_nei = []
    # for b in range (len(batch)) :
    #     _6_nei.append( batch[b]['6nn'][0] + index_offset )
    #     index_offset += batch[b]['6nn'][0].shape[0]
    # _6_nei = torch.cat (_6_nei )

    target_locs_hier = batch[0]['target_locs_hier']
    for h in range(len(target_locs_hier)):
        hier_locs = target_locs_hier[h]
        hier_locs = torch.cat([hier_locs, torch.zeros(hier_locs.shape[0], 1).long()], 1)
        for b in range(1, len(batch)):
            cur_locs = batch[b]['target_locs_hier'][h]
            cur_locs = torch.cat([cur_locs, torch.ones(cur_locs.shape[0], 1).long() * b], 1)
            hier_locs = torch.cat([hier_locs, cur_locs])
        target_locs_hier[h] = hier_locs

    for b in range(1, len(batch)):
        cur_locs = batch[b]['input'][0]
        cur_locs = torch.cat([cur_locs, torch.ones(cur_locs.shape[0], 1).long() * b], 1)
        locs = torch.cat([locs, cur_locs])
        feats_1 = torch.cat([feats_1, batch[b]['input'][1]])
        feats_2 = torch.cat([feats_2, batch[b]['input'][2]])

    motion = torch.cat([x['motion'] for x in batch])

    sdfs = torch.stack([x['sdf'] for x in batch])

    sdf_hierarchy = None
    if batch[0]['sdf_hierarchy'] is not None:
        sdf_hierarchy = [None] * len(batch[0]['sdf_hierarchy'])
        for h in range(len(batch[0]['sdf_hierarchy'])):
            sdf_hierarchy[h] = torch.stack([x['sdf_hierarchy'][h] for x in batch])

    mot_hierarchy = None
    if batch[0]['mot_hierarchy'] is not None:
        mot_hierarchy = [None] * len(batch[0]['mot_hierarchy'])
        for h in range(len(batch[0]['mot_hierarchy'])):
            mot_hierarchy[h] = torch.stack([x['mot_hierarchy'][h] for x in batch])

    return {'name': names,
            'input': [locs, feats_1, feats_2], 'motion': motion,
            'known': known,  # '6nn': _6_nei,
            'target_locs_hier': target_locs_hier,
            'sdf': sdfs,
            'sdf_hierarchy': sdf_hierarchy, 'mot_hierarchy': mot_hierarchy,
            'bsize': len(batch)
            }
