import numpy as np
import torch
import torch.nn.functional as F

import sparseconvnet as scn

import data_util

UNK_THRESH = 2
# UNK_THRESH = 3

UNK_ID = -1


def compute_targets(tgt_sdf, hierarchy_sdf,
                    num_hierarchy_levels, truncation, use_loss_masking, known):
    '''
    SDF values are rpecomputed during datagen for each level using depth fusion.
    convert SDF values to TSDF. this is done for each level
    compute Occupancise for output level, and max pooling for each sub level
    :param tgt_sdf: [B, 1, 96, 96, 128]
    :param hierarchy_sdf: 3 dense sdfs
    :param num_hierarchy_levels:
    :param truncation:
    :param use_loss_masking:
    :param known:
    :return: target_for_occs : 1: occupied, 0: empty, -1: unkown
    '''
    assert (len(tgt_sdf.shape) == 5)
    target_for_occs = [None] * num_hierarchy_levels
    target_sdf_hier = [None] * num_hierarchy_levels
    target_for_sdf = data_util.preprocess_sdf_pt(tgt_sdf, truncation)
    known_mask = None
    target_sdf_hier[-1] = target_for_sdf.clone()
    target_occ = (torch.abs(target_for_sdf) < truncation).float()
    # if use_loss_masking:
    #     target_occ[known >= UNK_THRESH] = UNK_ID
    target_for_occs[-1] = target_occ

    factor = 2
    for h in range(num_hierarchy_levels - 2, -1, -1):
        target_for_occs[h] = torch.nn.MaxPool3d(kernel_size=2)(target_for_occs[h + 1])
        if hierarchy_sdf[h] is not None:
            target_sdf_hier[h] = data_util.preprocess_sdf_pt(hierarchy_sdf[h], truncation)
        else:
            target_sdf_hier[h] = None
        factor *= 2
    return target_for_sdf, target_for_occs, target_sdf_hier


# note: weight_missing_geo must be > 1
def compute_weights_missing_geo(weight_missing_geo, input_locs, target_for_occs, truncation):
    '''
    Compute the weigths for voxels,
    The weigth values are set to 1 for input locations, and 5 for other locations
    :param weight_missing_geo: weight of missing geometry
    :param input_locs: non-zeros locations in the input [nnz, 4] 4--> X,Y,Z,Batch_ID
    :param target_for_occs:
    :param truncation:
    :return:
    '''
    # import matplotlib.pyplot as plt
    # plt.imshow(weights[-1][0][0][60].cpu().numpy())
    # weights[-1][0][0][60].cpu().numpy()

    num_hierarchy_levels = len(target_for_occs)
    weights = [None] * num_hierarchy_levels
    dims = target_for_occs[-1].shape[2:]
    flatlocs = input_locs[:, 3] * dims[0] * dims[1] * dims[2] + input_locs[:, 0] * dims[1] * dims[2] + input_locs[:,
                                                                                                       1] * dims[
                   2] + input_locs[:, 2]
    weights[-1] = torch.ones(target_for_occs[-1].shape, dtype=torch.int32).cuda()  # init all weights as 1
    weights[-1].view(-1)[flatlocs] += 1  # +1 for input localtions
    weights[-1][torch.abs(target_for_occs[-1]) <= truncation] += 3  # +3 for all locations
    weights[-1] = (weights[-1] == 4).float() * (weight_missing_geo - 1) + 1  # Missing region:5 input region: 1
    factor = 2
    for h in range(num_hierarchy_levels - 2, -1, -1):
        weights[h] = weights[h + 1][:, :, ::2, ::2, ::2].contiguous()
        factor *= 2
    return weights


def apply_log_transform(sdf):
    sgn = torch.sign(sdf)
    out = torch.log(torch.abs(sdf) + 1)
    out = sgn * out
    return out


def compute_bce_sparse_dense(
        sparse_pred_locs, sparse_pred_vals, dense_tgts, weights, use_loss_masking, truncation=3, batched=True):
    assert (len(dense_tgts.shape) == 5 and dense_tgts.shape[1] == 1)
    dims = dense_tgts.shape[2:]
    loss = 0.0 if batched else np.zeros(dense_tgts.shape[0], dtype=np.float32)

    predvalues = sparse_pred_vals.view(-1)
    flatlocs = sparse_pred_locs[:, 3] * dims[0] * dims[1] * dims[2] + sparse_pred_locs[:, 0] * dims[1] * dims[
        2] + sparse_pred_locs[:, 1] * dims[2] + sparse_pred_locs[:, 2]
    tgtvalues = dense_tgts.view(-1)[flatlocs]
    weight = None if weights is None else weights.view(-1)[flatlocs]
    if use_loss_masking:
        mask = tgtvalues != UNK_ID
        tgtvalues = tgtvalues[mask]
        predvalues = predvalues[mask]
        if weight is not None:
            weight = weight[mask]
    else:
        tgtvalues[tgtvalues == UNK_ID] = 0
    if batched:
        loss = F.binary_cross_entropy_with_logits(predvalues, tgtvalues, weight=weight)
    else:
        if dense_tgts.shape[0] == 1:
            loss[0] = F.binary_cross_entropy_with_logits(predvalues, tgtvalues, weight=weight)
        else:
            raise NotImplementedError()
    return loss


def compute_l2_flow_sparse_dense(
        sparse_pred_locs, sparse_pred_flow, dense_tgt_flow, weights, use_loss_masking, truncation=3, batched=True):
    assert (len(dense_tgt_flow.shape) == 5 and dense_tgt_flow.shape[-1] == 3)
    dims = dense_tgt_flow.shape[1:4]
    loss = 0.0 if batched else np.zeros(dense_tgt_flow.shape[0], dtype=np.float32)

    predvalues = sparse_pred_flow.view(-1, 3)
    flatlocs = sparse_pred_locs[:, 3] * dims[0] * dims[1] * dims[2] + sparse_pred_locs[:, 0] * dims[1] * dims[
        2] + sparse_pred_locs[:, 1] * dims[2] + sparse_pred_locs[:, 2]
    tgtvalues = dense_tgt_flow.view(-1, 3)[flatlocs]
    weight = None if weights is None else weights.view(-1)[flatlocs]

    if use_loss_masking:
        mask = tgtvalues != UNK_ID
        tgtvalues = tgtvalues[mask]
        predvalues = predvalues[mask]
        if weight is not None:
            weight = weight[mask]
    else:
        tgtvalues[tgtvalues == UNK_ID] = 0

    """apply loss on voxel psoition where GT is available"""
    flow_mask = ~torch.isinf(tgtvalues[:, 0])
    tgtvalues = tgtvalues[flow_mask]
    predvalues = predvalues[flow_mask]
    weight = None if weight is None else weight[flow_mask]

    if batched:
        if weight is None:
            weight = 1.
        loss = torch.mean(weight * torch.sum((predvalues - tgtvalues) ** 2, 1) / 2.0)
    else:
        raise NotImplementedError()

    return loss


def compute_iou_sparse_dense(sparse_pred_locs, dense_tgts, use_loss_masking, truncation=3, batched=True):
    assert (len(dense_tgts.shape) == 5 and dense_tgts.shape[1] == 1)
    dims = dense_tgts.shape[2:]
    corr = 0.0 if batched else np.zeros(dense_tgts.shape[0], dtype=np.float32)
    union = 0.0 if batched else np.zeros(dense_tgts.shape[0], dtype=np.float32)
    for b in range(dense_tgts.shape[0]):
        tgt = dense_tgts[b, 0]
        if sparse_pred_locs[b] is None:
            continue
        predlocs = sparse_pred_locs[b]
        # flatten locs # TODO not sure whats the most efficient way to compute this...
        predlocs = predlocs[:, 0] * dims[1] * dims[2] + predlocs[:, 1] * dims[2] + predlocs[:, 2]
        tgtlocs = torch.nonzero(tgt == 1)
        tgtlocs = tgtlocs[:, 0] * dims[1] * dims[2] + tgtlocs[:, 1] * dims[2] + tgtlocs[:, 2]
        if use_loss_masking:
            tgtlocs = tgtlocs.cpu().numpy()
            # mask out from pred
            mask = torch.nonzero(tgt == UNK_ID)
            mask = mask[:, 0] * dims[1] * dims[2] + mask[:, 1] * dims[2] + mask[:, 2]
            predlocs = predlocs.cpu().numpy()
            if mask.shape[0] > 0:
                _, mask, _ = np.intersect1d(predlocs, mask.cpu().numpy(), return_indices=True)
                predlocs = np.delete(predlocs, mask)
        else:
            predlocs = predlocs.cpu().numpy()
            tgtlocs = tgtlocs.cpu().numpy()
        if batched:
            corr += len(np.intersect1d(predlocs, tgtlocs, assume_unique=True))
            union += len(np.union1d(predlocs, tgtlocs))
        else:
            corr[b] = len(np.intersect1d(predlocs, tgtlocs, assume_unique=True))
            union[b] = len(np.union1d(predlocs, tgtlocs))
    if not batched:
        return np.divide(corr, union)
    if union > 0:
        return corr / union
    return -1


def compute_l1_predsurf_sparse_dense(sparse_pred_locs, sparse_pred_vals, dense_tgts, weights, use_log_transform,
                                     use_loss_masking, known, batched=True, thresh=None):
    assert (len(dense_tgts.shape) == 5 and dense_tgts.shape[1] == 1)
    dims = dense_tgts.shape[2:]
    loss = 0.0 if batched else np.zeros(dense_tgts.shape[0], dtype=np.float32)

    locs = sparse_pred_locs if thresh is None else sparse_pred_locs[sparse_pred_vals.view(-1) <= thresh]
    predvalues = sparse_pred_vals.view(-1) if thresh is None else sparse_pred_vals.view(-1)[
        sparse_pred_vals.view(-1) <= thresh]
    flatlocs = locs[:, 3] * dims[0] * dims[1] * dims[2] + locs[:, 0] * dims[1] * dims[2] + locs[:, 1] * dims[2] + locs[
                                                                                                                  :, 2]
    tgtvalues = dense_tgts.view(-1)[flatlocs]
    weight = None if weights is None else weights.view(-1)[flatlocs]
    if use_loss_masking:
        mask = known < UNK_THRESH
        mask = mask.view(-1)[flatlocs]
        predvalues = predvalues[mask]
        tgtvalues = tgtvalues[mask]
        if weight is not None:
            weight = weight[mask]
    if use_log_transform:
        predvalues = apply_log_transform(predvalues)
        tgtvalues = apply_log_transform(tgtvalues)
    if batched:
        if weight is not None:
            loss = torch.abs(predvalues - tgtvalues)
            loss = torch.mean(loss * weight)
        else:
            loss = torch.mean(torch.abs(predvalues - tgtvalues))
    else:
        if dense_tgts.shape[0] == 1:
            if weight is not None:
                loss_ = torch.abs(predvalues - tgtvalues)
                loss[0] = torch.mean(loss_ * weight).item()
            else:
                loss[0] = torch.mean(torch.abs(predvalues - tgtvalues)).item()
        else:
            raise NotImplementedError()
    return loss


def arap_loss(locs, pred_trn, pred_rot, _6_nei, input_dim):
    ''' Energy_ARAP = ||R_0(V_0 - V_j) -  ( V_0 + T_0 - V_j - T_j )||^2 _{j=1~k}
    :param locs:
    :param input_trn: voxel-metric
    :param input_mask:
    :param pred_trn: voxel-metric
    :param pred_rot:
    :param input_dim:
    :return:
    '''

    dimx, dimy, dimz = input_dim

    # xyz_t = xyz.permute(0, 2, 1).contiguous()
    #
    # B, N, C = xyz_t.shape
    #
    # distance, idx = pointutils.knn( KNN + 1 , xyz_t, xyz_t)
    #
    # p3d_j_0 = pointutils.grouping_operation(xyz, idx)[:,:,:,1:]  # [B, 3, N, S]
    # # print( "trn.is_contiguous()", trn.is_contiguous())
    # # print( "trn", trn.shape)
    # # print( "idx.is_contiguous()", idx.is_contiguous())
    # trn_j = pointutils.grouping_operation(trn.contiguous(), idx)[:,:,:,1:]
    #
    # trn_i = trn.view(B,3,-1, 1)
    # rot_i = rot
    # p3d_i_0 = xyz.view(B,3,-1, 1)

    N = locs.shape[0]

    p3d_i_0 = locs[:, :3].view(-1, 1, 3)  # [N,3]
    trn_i = pred_trn.view(-1, 1, 3)  # [N,3]

    p3d_j_0 = locs[:, :3][_6_nei.view(-1)].view(N, 6, 3)  # [N,6,3]
    trn_j = pred_trn[_6_nei.view(-1)].view(N, 6, 3)  # [N,6,3]

    p3d_i_1 = p3d_i_0 + trn_i
    p3d_j_1 = p3d_j_0 + trn_j
    dis_0 = p3d_i_0 - p3d_j_0  # [N,6,3]
    dis_1 = p3d_i_1 - p3d_j_1  # [N,6,3]

    dis_1 = dis_1.view(-1, 6, 3, 1)
    rot_i = pred_rot.view(-1, 1, 3, 3)
    rot_dis_1 = torch.matmul(rot_i, dis_1)

    residue = dis_0 - rot_dis_1[:, :, :, 0]  # [N,6,3]
    mask = (_6_nei > 0).float()
    arap_loss = torch.mean(torch.sum(mask * torch.sum(residue ** 2, 2) / 2.0, 1))

    return arap_loss


def l2_flow_norm(pred, target):
    '''
    :param pred: [n, 3]
    :param target: [n, 3]
    :return:
    '''
    loss = torch.mean(torch.sum((pred - target) ** 2, 1) / 2.0)
    return loss


def euclidean_distance(pred, target):
    '''
    :param pred: [n,3]
    :param target: [n,3]
    :return:
    '''
    epe = torch.mean(torch.sqrt(torch.sum((pred - target) ** 2, 1)))
    return epe


def cosine_similarity(pred, target):
    '''
    :param pred: [n,3]
    :param target: [n,3]
    :return:
    '''

    pred_norm = torch.norm(pred, dim=1, keepdim=True)
    target_norm = torch.norm(target, dim=1, keepdim=True)

    unit_pred = pred / (pred_norm + 1e-10)
    unit_target = target / (target_norm + 1e-10)

    similarity_loss = torch.mean(1 - torch.sum(unit_pred * unit_target, 1))

    return similarity_loss


def compute_loss(
        output_4D_field, output_hier,
        target_for_sdf, target_for_occs, target_sdf_hier,
        target_for_mot, target_mot_hier,
        loss_weights, mot_loss_weight, truncation, use_log_transform=True, weight_missing_geo=1,
        input_locs=None, use_loss_masking=True, known=None, batched=True):
    assert (len(output_hier) == len(target_for_occs))
    batch_size = target_for_sdf.shape[0]
    loss = 0.0 if batched else np.zeros(batch_size, dtype=np.float32)
    geo_losses = [] if batched else [[] for i in range(len(output_hier) + 1)]
    mot_losses = [] if batched else [[] for i in range(len(output_hier) + 1)]

    weights = [None] * len(target_for_occs)
    if weight_missing_geo > 1:
        weights = compute_weights_missing_geo(weight_missing_geo, input_locs, target_for_occs, truncation)
    for h in range(len(output_hier)):
        # output_hier_field : [num, 5 (occ, sdf, mot(3))]
        output_hier_locs, output_hier_field = output_hier[h]

        if len(output_hier_locs) == 0 or loss_weights[h] == 0:
            if batched:
                geo_losses.append(-1)
                mot_losses.append(-1)
            else:
                geo_losses[h].extend([-1] * batch_size)
                mot_losses[h].extend([-1] * batch_size)
            continue

        output_hier_occ = output_hier_field[:, 0]
        output_hier_sdf = output_hier_field[:, 1]
        output_hier_mot = output_hier_field[:, 2:]

        cur_loss_occ = compute_bce_sparse_dense(
            output_hier_locs, output_hier_occ, target_for_occs[h], weights[h], use_loss_masking, batched=batched)

        cur_known = None if not use_loss_masking else (target_for_occs[h] == UNK_ID) * UNK_THRESH

        cur_loss_sdf = compute_l1_predsurf_sparse_dense(
            output_hier_locs, output_hier_sdf, target_sdf_hier[h], weights[h], use_log_transform, use_loss_masking,
            cur_known, batched=batched)

        cur_loss_mot = compute_l2_flow_sparse_dense(
            output_hier_locs, output_hier_mot, target_mot_hier[h][:, 0], weights[h], use_loss_masking, batched=batched)

        cur_geo_loss = cur_loss_occ + cur_loss_sdf
        cur_mot_loss = mot_loss_weight[h] * cur_loss_mot

        if batched:
            loss += loss_weights[h] * (cur_geo_loss + cur_mot_loss)
            geo_losses.append(cur_geo_loss.item())
            mot_losses.append(cur_mot_loss.item())
        else:
            loss += loss_weights[h] * (cur_geo_loss + cur_mot_loss)
            geo_losses[h].extend(cur_geo_loss)
            mot_losses[h].extend(cur_mot_loss)

    # loss on surface
    if len(output_4D_field[0]) > 0 and loss_weights[-1] > 0:

        output_surf_locs, output_surf_field = output_4D_field

        output_surf_sdf = output_surf_field[:, 0]
        output_surf_mot = output_surf_field[:, 1:]

        cur_loss_sdf = compute_l1_predsurf_sparse_dense(
            output_surf_locs, output_surf_sdf, target_for_sdf, weights[-1], use_log_transform, use_loss_masking, known,
            batched=batched)

        cur_loss_mot = compute_l2_flow_sparse_dense(
            output_surf_locs, output_surf_mot, target_for_mot[:, 0], weights[-1], use_loss_masking, batched=batched)

        cur_geo_loss = cur_loss_sdf
        cur_mot_loss = mot_loss_weight[-1] * cur_loss_mot

        if batched:
            loss += loss_weights[-1] * (cur_geo_loss + cur_mot_loss)
            geo_losses.append(cur_geo_loss.item())
            mot_losses.append(cur_mot_loss.item())
        else:
            loss += loss_weights[-1] * (cur_geo_loss + cur_mot_loss)
            geo_losses[len(output_hier)].extend(cur_geo_loss)
            mot_losses[len(output_hier)].extend(cur_mot_loss)

    else:
        if batched:
            geo_losses.append(-1)
            mot_losses.append(-1)
        else:
            geo_losses[len(output_hier)].extend([-1] * batch_size)
            mot_losses[len(output_hier)].extend([-1] * batch_size)

    return loss, geo_losses, mot_losses


def compute_shape_loss(
        output_distance_field, output_hier,
        target_for_sdf, target_for_occs, target_sdf_hier,
        loss_weights, truncation, use_log_transform=True, weight_missing_geo=1,
        input_locs=None, use_loss_masking=True, known=None, batched=True):
    assert (len(output_hier) == len(target_for_occs))
    batch_size = target_for_sdf.shape[0]
    loss = 0.0
    geo_losses = []

    weights = [None] * len(target_for_occs)
    if weight_missing_geo > 1:
        weights = compute_weights_missing_geo(weight_missing_geo, input_locs, target_for_occs, truncation)
    for h in range(len(output_hier)):
        # output_hier_field : [num, 2 (occ, sdf)]
        output_hier_locs, output_hier_field = output_hier[h]

        if len(output_hier_locs) == 0 or loss_weights[h] == 0:
            geo_losses.append(-1)
            continue

        output_hier_occ = output_hier_field[:, 0]
        output_hier_sdf = output_hier_field[:, 1]

        cur_loss_occ = compute_bce_sparse_dense(
            output_hier_locs, output_hier_occ, target_for_occs[h], weights[h], use_loss_masking, batched=batched)

        cur_known = None if not use_loss_masking else (target_for_occs[h] == UNK_ID) * UNK_THRESH

        if target_sdf_hier[h] is not None:
            cur_loss_sdf = compute_l1_predsurf_sparse_dense(
                output_hier_locs, output_hier_sdf, target_sdf_hier[h], weights[h], use_log_transform, use_loss_masking,
                cur_known, batched=batched)

        else:
            cur_loss_sdf = 0

        cur_geo_loss = cur_loss_occ + cur_loss_sdf

        loss += loss_weights[h] * cur_geo_loss
        geo_losses.append(cur_geo_loss.item())

    if len(output_distance_field[0]) > 0 and loss_weights[-1] > 0:

        output_surf_locs, output_surf_field = output_distance_field

        output_surf_sdf = output_surf_field[:, 0]

        cur_loss_sdf = compute_l1_predsurf_sparse_dense(
            output_surf_locs, output_surf_sdf, target_for_sdf, weights[-1], use_log_transform, use_loss_masking, known,
            batched=batched)

        cur_geo_loss = cur_loss_sdf

        loss += loss_weights[-1] * cur_geo_loss
        geo_losses.append(cur_geo_loss.item())


    else:

        geo_losses.append(-1)

    return loss, geo_losses


def compute_l1_tgtsurf_sparse_dense(sparse_pred_locs, sparse_pred_vals, dense_tgts, truncation, use_loss_masking, known,
                                    batched=True, thresh=None):
    assert (len(dense_tgts.shape) == 5 and dense_tgts.shape[1] == 1)
    batch_size = dense_tgts.shape[0]
    dims = dense_tgts.shape[2:]
    loss = 0.0 if batched else np.zeros(dense_tgts.shape[0], dtype=np.float32)
    pred_dense = torch.ones(batch_size * dims[0] * dims[1] * dims[2]).to(dense_tgts.device)
    fill_val = -truncation
    pred_dense.fill_(fill_val)
    if thresh is not None:
        tgtlocs = torch.nonzero(torch.abs(dense_tgts) <= thresh)
    else:
        tgtlocs = torch.nonzero(torch.abs(dense_tgts) < truncation)
    batchids = tgtlocs[:, 0]
    tgtlocs = tgtlocs[:, 0] * dims[0] * dims[1] * dims[2] + tgtlocs[:, 2] * dims[1] * dims[2] + tgtlocs[:, 3] * dims[
        2] + tgtlocs[:, 4]
    tgtvalues = dense_tgts.view(-1)[tgtlocs]
    flatlocs = sparse_pred_locs[:, 3] * dims[0] * dims[1] * dims[2] + sparse_pred_locs[:, 0] * dims[1] * dims[
        2] + sparse_pred_locs[:, 1] * dims[2] + sparse_pred_locs[:, 2]
    pred_dense[flatlocs] = sparse_pred_vals.view(-1)
    predvalues = pred_dense[tgtlocs]
    if use_loss_masking:
        mask = known < UNK_THRESH
        mask = mask.view(-1)[tgtlocs]
        tgtvalues = tgtvalues[mask]
        predvalues = predvalues[mask]
    if batched:
        loss = torch.mean(torch.abs(predvalues - tgtvalues)).item()
    else:
        if dense_tgts.shape[0] == 1:
            loss[0] = torch.mean(torch.abs(predvalues - tgtvalues)).item()
        else:
            raise NotImplementedError()

    return loss
