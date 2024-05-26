import numpy as np
import torch
import torch.nn as nn

import sparseconvnet as scn
from sgnn.sgnn_model import SparseEncoderLayer

def count_num_model_params(model):
    num = 0
    for p in list(model.parameters()):
        cur = 1
        for s in list(p.size()):
            cur = cur * s
        num += cur
    return num

FSIZE0 = 3
FSIZE1 = 2


class _4DEncoder(nn.Module):
    def __init__(self, nf_in, nf_per_level, nf_out, use_skip_sparse, use_skip_dense, log_IO, input_volume_size):
        nn.Module.__init__(self)
        assert (type(nf_per_level) is list)
        data_dim = 3
        self.use_skip_sparse = use_skip_sparse
        self.use_skip_dense = use_skip_dense
        # self.use_bias = True
        self.use_bias = False
        modules = []
        volume_sizes = [(np.array(input_volume_size) // (k + 1)).tolist() for k in range(len(nf_per_level))]
        for level in range(len(nf_per_level)):
            nf_in = nf_in if level == 0 else nf_per_level[level - 1]
            input_sparsetensor = level > 0
            return_sparsetensor = (level < len(nf_per_level) - 1)
            modules.append(SparseEncoderLayer(nf_in, nf_per_level[level], input_sparsetensor, return_sparsetensor,
                                              volume_sizes[level]))
        self.process_sparse = nn.Sequential(*modules)
        nf = nf_per_level[-1]
        # 16 -> 8
        nf0 = nf * 3 // 2
        self.encode_dense0 = nn.Sequential(
            nn.Conv3d(nf, nf0, kernel_size=4, stride=2, padding=1, bias=self.use_bias),
            nn.BatchNorm3d(nf0),
            nn.ReLU(True)
        )
        # 8 -> 4
        nf1 = nf * 2
        self.encode_dense1 = nn.Sequential(
            nn.Conv3d(nf0, nf1, kernel_size=4, stride=2, padding=1, bias=self.use_bias),
            nn.BatchNorm3d(nf1),
            nn.ReLU(True)
        )
        # 4 -> 4
        nf2 = nf1
        self.bottleneck_dense2 = nn.Sequential(
            nn.Conv3d(nf1, nf2, kernel_size=1, bias=self.use_bias),
            nn.BatchNorm3d(nf2),
            nn.ReLU(True)
        )
        # 4 -> 8
        nf3 = nf2 if not self.use_skip_dense else nf1 + nf2
        nf4 = nf3 // 2
        self.decode_dense3 = nn.Sequential(
            nn.ConvTranspose3d(nf3, nf4, kernel_size=4, stride=2, padding=1, bias=self.use_bias),
            nn.BatchNorm3d(nf4),
            nn.ReLU(True)
        )
        # 8 -> 16
        if self.use_skip_dense:
            nf4 += nf0
        nf5 = nf4 // 2
        self.decode_dense4 = nn.Sequential(
            nn.ConvTranspose3d(nf4, nf5, kernel_size=4, stride=2, padding=1, bias=self.use_bias),
            nn.BatchNorm3d(nf5),
            nn.ReLU(True)
        )
        self.final = nn.Sequential(
            nn.Conv3d(nf5, nf_out, kernel_size=1, bias=self.use_bias),
            nn.BatchNorm3d(nf_out),
            nn.ReLU(True)
        )
        # occ prediction
        self.occpred = nn.Sequential(
            nn.Conv3d(nf_out, 1, kernel_size=1, bias=self.use_bias)
        )
        # distance field prediction
        self.sdfpred = nn.Sequential(
            nn.Conv3d(nf_out, 1, kernel_size=1, bias=self.use_bias)
        )

        # debug stats
        params_encodesparse = count_num_model_params(self.process_sparse)
        params_encodedense = count_num_model_params(self.encode_dense0) + count_num_model_params(
            self.encode_dense0) + count_num_model_params(self.encode_dense1) + count_num_model_params(
            self.bottleneck_dense2)
        params_decodedense = count_num_model_params(self.decode_dense3) + count_num_model_params(
            self.decode_dense4) + count_num_model_params(self.final) + count_num_model_params(self.occpred)
        print('[TSDFEncoder] params encode sparse', params_encodesparse)
        print('[TSDFEncoder] params encode dense', params_encodedense)
        print('[TSDFEncoder] params decode dense', params_decodedense)

    def forward(self, x, pred_shape= True):
        feats_sparse = []
        for k in range(len(self.process_sparse)):
            x, ft = self.process_sparse[k](x)
            if self.use_skip_sparse:
                feats_sparse.extend(ft)

        enc0 = self.encode_dense0(x)
        enc1 = self.encode_dense1(enc0)
        bottleneck = self.bottleneck_dense2(enc1)
        if self.use_skip_dense:
            dec0 = self.decode_dense3(torch.cat([bottleneck, enc1], 1))
        else:
            dec0 = self.decode_dense3(bottleneck)
        if self.use_skip_dense:
            x = self.decode_dense4(torch.cat([dec0, enc0], 1))
        else:
            x = self.decode_dense4(dec0)
        x = self.final(x)

        shape_out = None
        if pred_shape :
            occ = self.occpred(x)
            sdf = self.sdfpred(x)
            shape_out = torch.cat([occ, sdf], 1)

        return x, shape_out, feats_sparse

class ShapeDecoder(nn.Module):
    def __init__(self, nf_in, nf, pass_occ, pass_feats, max_data_size, truncation=3):
        nn.Module.__init__(self)
        data_dim = 3
        self.pass_occ = pass_occ
        self.pass_feats = pass_feats
        self.nf_in = nf_in
        self.nf = nf
        self.truncation = truncation
        self.p0 = scn.InputLayer(data_dim, max_data_size, mode=0)
        self.p1 = scn.SubmanifoldConvolution(data_dim, nf_in, nf, filter_size=FSIZE0, bias=False)
        self.p2 = scn.FullyConvolutionalNet(data_dim, reps=1, nPlanes=[nf, nf, nf], residual_blocks=True)
        self.p3 = scn.BatchNormReLU(nf * 3)
        self.p4 = scn.OutputLayer(data_dim)

        # upsampled
        self.n0 = scn.InputLayer(data_dim, max_data_size, mode=0)
        self.n1 = scn.SubmanifoldConvolution(data_dim, nf * 3, nf, filter_size=FSIZE0, bias=False)
        self.n2 = scn.BatchNormReLU(nf)
        self.n3 = scn.OutputLayer(data_dim)
        self.linear = nn.Linear(nf, 1)
        self.linearsdf = nn.Linear(nf, 1)


    def to_next_level_locs(self, locs, feats):  # upsample factor of 2 predictions
        assert (len(locs.shape) == 2)
        data_dim = locs.shape[-1] - 1  # assumes batch mode
        offsets = torch.nonzero(torch.ones(2, 2, 2)).long()  # 8 x 3
        locs_next = locs.unsqueeze(1).repeat(1, 8, 1)
        locs_next[:, :, :data_dim] *= 2
        locs_next[:, :, :data_dim] += offsets
        # print('locs', locs.shape, locs.type())
        # print('locs_next', locs_next.shape, locs_next.type())
        # print('locs_next.view(-1,4)[:20]', locs_next.view(-1,4)[:20])
        feats_next = feats.unsqueeze(1).repeat(1, 8, 1)  # TODO: CUSTOM TRILERP HERE???
        # print('feats', feats.shape, feats.type())
        # print('feats_next', feats_next.shape, feats_next.type())
        # print('feats_next.view(-1,feats.shape[-1])[:20,:5]', feats_next.view(-1,feats.shape[-1])[:20,:5])
        # raw_input('sdlfkj')
        return locs_next.view(-1, locs.shape[-1]), feats_next.view(-1, feats.shape[-1])

    def forward(self, x):
        '''
        :param x:
        :return:  masked [locs, feature], raw [locs, pred]
        '''
        input_locs = x[0]
        if len(input_locs) == 0:
            return [[], []], [[], []]
        # x=self.sparseModel(x)
        # print('x(sparse)', x.shape)

        x = self.p0(x)
        x = self.p1(x)
        x = self.p2(x)
        x = self.p3(x)
        x = self.p4(x)

        locs_unfilt, feats = self.to_next_level_locs(input_locs, x)

        x = self.n0([locs_unfilt, feats])
        x = self.n1(x)
        x = self.n2(x)
        x = self.n3(x)

        # predict occupancy
        out = self.linear(x)
        sdf = self.linearsdf(x)
        # mot = self.linearmot(x)
        # mask out for next level processing
        mask = (nn.Sigmoid()(out) > 0.5).view(-1)
        # print('x', x.type(), x.shape, torch.min(x).item(), torch.max(x).item())
        # print('locs_unfilt', locs_unfilt.type(), locs_unfilt.shape, torch.min(locs_unfilt).item(), torch.max(locs_unfilt).item())
        # print('out', out.type(), out.shape, torch.min(out).item(), torch.max(out).item())
        # print('mask', mask.type(), mask.shape, torch.sum(mask).item())
        locs = locs_unfilt[mask]

        out = torch.cat([out, sdf], 1)
        if self.pass_feats and self.pass_occ:
            feats = torch.cat([x[mask], out[mask]], 1)
        elif self.pass_feats:
            feats = x[mask]
        elif self.pass_occ:
            feats = out[mask]
        return [locs, feats], [locs_unfilt, out]

class SurfacePrediction(nn.Module):
    def __init__(self, nf_in, nf,  max_data_size):
        nn.Module.__init__(self)
        data_dim = 3
        self.p0 = scn.InputLayer(data_dim, max_data_size, mode=0)
        self.p1 = scn.SubmanifoldConvolution(data_dim, nf_in, nf, filter_size=FSIZE0, bias=False)
        self.p2 = scn.FullyConvolutionalNet(data_dim, reps=1, nPlanes=[nf, nf, nf],
                                            residual_blocks=True)  # nPlanes=[nf, nf*2, nf*2], residual_blocks=True)
        self.p3 = scn.BatchNormReLU(nf * 3)
        self.p4 = scn.OutputLayer(data_dim)
        self.sdflinear = nn.Linear(nf *3, 1)

    def forward(self, x):
        if len(x[0]) == 0:
            return [], []
        # x=self.sparseModel(x)
        # print('x(sparse)', x.shape)

        x = self.p0(x)
        x = self.p1(x)
        x = self.p2(x)
        x = self.p3(x)
        x = self.p4(x)

        sdf = self.sdflinear(x)
        return sdf

class MotionShapeCompleteModel(nn.Module):
    def __init__(self,  input_dim, pass_occ, pass_feats, use_skip_sparse,
                 use_skip_dense, log_IO, truncation=3):
        nn.Module.__init__(self)

        self.data_dim = 3

        encoder_dim = 16
        self.nf_per_level = [encoder_dim, int(encoder_dim * 1.5), encoder_dim * 2]  # 20 30 40
        nf_coarse = 32

        nf_shape_decoder = 16
        num_hierarchy_levels = 4

        input_nf = 4

        self.truncation = truncation
        self.pass_occ = pass_occ
        self.pass_feats = pass_feats
        # encoder
        if not isinstance(input_dim, (list, tuple, np.ndarray)):
            input_dim = [input_dim, input_dim, input_dim]

        # self.nf_per_level = [encoder_dim, int(encoder_dim * 1.5), encoder_dim * 2]

        self.use_skip_sparse = use_skip_sparse
        self.encoder = _4DEncoder(
            input_nf, self.nf_per_level, nf_coarse,
            self.use_skip_sparse, use_skip_dense, log_IO, input_volume_size=input_dim)

        self.refine_sizes = [(np.array(input_dim) // (pow(2,k))).tolist() for k in range(num_hierarchy_levels-1)][::-1]
        self.nf_per_level.append(self.nf_per_level[-1])
        log_IO.write_to_terminal('#params encoder: ' + str( count_num_model_params(self.encoder)))

        '''sparse shape prediction'''
        self.shape_decoder = scn.Sequential()
        for h in range(1, num_hierarchy_levels):
            nf_in = 0 if not self.use_skip_sparse else self.nf_per_level[num_hierarchy_levels - h]
            if pass_occ:
                nf_in += 2 #  (occ + sdf) for sgnn
            if pass_feats:
                nf_in += (nf_coarse if h == 1 else nf_shape_decoder)
                # nf_in += (nf_coarse if h == 1 else self.nf_per_level[num_hierarchy_levels - h])
            self.shape_decoder.add(ShapeDecoder(nf_in, nf_shape_decoder, pass_occ, pass_feats, self.refine_sizes[h - 1], truncation=self.truncation))

        '''predict fineset surface'''
        nf_in = 0 if not self.use_skip_sparse else self.nf_per_level[0]
        if pass_occ:
            nf_in += 2
        if pass_feats:
            nf_in += nf_shape_decoder
        self.surfacepred = SurfacePrediction(nf_in, nf_shape_decoder, self.refine_sizes[-1])
        log_IO.write_to_terminal('#params shape decoder: ' +
                                 str(count_num_model_params(self.shape_decoder) + count_num_model_params(self.surfacepred)))

        '''sparse motion prediction'''
        self.mot_decoder = scn.Sequential()
        for h in range(1, num_hierarchy_levels):
            nf_in = 0 if not self.use_skip_sparse else self.nf_per_level[num_hierarchy_levels - h]
            if pass_feats:
                nf_in += (nf_coarse if h == 1 else self.nf_per_level[num_hierarchy_levels - h + 1])
            # if self.encode_region_mask:
            #     nf_in += 1
            self.mot_decoder.add(
                MotDecoder(nf_in, self.nf_per_level[num_hierarchy_levels - h], pass_occ, pass_feats,
                           self.refine_sizes[h - 1], truncation=self.truncation))
        log_IO.write_to_terminal('#params MotDecoder:' + str( count_num_model_params(self.mot_decoder)))
        self.PRED_MOT = True
        if self.PRED_MOT:
            nf_in = 0 if not self.use_skip_sparse else self.nf_per_level[0]
            if pass_feats:
                nf_in += self.nf_per_level[1]
            # if self.encode_region_mask:
            #     nf_in += 1
            self.motion_pred = MotionPrediction(nf_in, self.nf_per_level[0], self.refine_sizes[-1])
            log_IO.write_to_terminal('#params motion_pred' + str( count_num_model_params(self.motion_pred)) )

    def dense_coarse_to_sparse(self, coarse_feats, coarse_occ, truncation):
        ''' convert dense coarse feature to sparse feature
        :param coarse_feats:
        :param coarse_occ: [B, 5 (occ + sdf + flow), 12 (dimx), 12 (dimy), 16 (dimz)]
        :param truncation:
        :return: locs, feats, [locs_unfilt, coarse_occ.view(-1, 5)]
                _, coarse_feats + coarse_occ
        '''
        nf = coarse_feats.shape[1]
        batch_size = coarse_feats.shape[0]
        # sparse locations
        locs_unfilt = torch.nonzero(torch.ones([coarse_occ.shape[2], coarse_occ.shape[3], coarse_occ.shape[4]])).unsqueeze(0).repeat(coarse_occ.shape[0], 1, 1).view(-1, 3)
        batches = torch.arange(coarse_occ.shape[0]).to(locs_unfilt.device).unsqueeze(1).repeat(1, coarse_occ.shape[2]*coarse_occ.shape[3]*coarse_occ.shape[4]).view(-1, 1)
        locs_unfilt = torch.cat([locs_unfilt, batches], 1)
        mask = nn.Sigmoid()(coarse_occ[:,0,:,:,:]) > 0.5
        if self.pass_feats:
            feats_feats = coarse_feats.permute(0, 2, 3, 4, 1).contiguous().view(batch_size, -1, nf)
            feats_feats = feats_feats[mask.view(batch_size, -1)]
        coarse_occ = coarse_occ.permute(0, 2, 3, 4, 1).contiguous()
        if self.pass_occ:
            occ_feats = coarse_occ[mask]
        if self.pass_occ and self.pass_feats:
            feats = torch.cat([occ_feats, feats_feats], 1)
        elif self.pass_occ:
            feats = occ_feats
        elif self.pass_feats:
            feats = feats_feats
        locs = locs_unfilt[mask.view(-1)]
        return locs, feats, [locs_unfilt, coarse_occ.view(-1, 2)]  # _, _,

    def indexing_dense_to_sparse(self, dense_feats, sparse_locs):

        B, nf, dx, dy, dz = dense_feats.shape
        dims = [dx, dy, dz]

        flatlocs = sparse_locs[:, 3] * dims[0] * dims[1] * dims[2] + \
                   sparse_locs[:, 0] * dims[1] * dims[2] + \
                   sparse_locs[:, 1] * dims[2] + \
                   sparse_locs[:, 2]

        dense_feats = dense_feats.permute( 0,2,3,4,1).contiguous().view(-1, nf)
        sparse_feats = dense_feats[flatlocs]

        return  sparse_locs, sparse_feats

    def concat_skip(self, x_from, x_to, spatial_size, batch_size):

        locs_from = x_from[0]
        locs_to = x_to[0]

        if len(locs_from) == 0 or len(locs_to) == 0:
            return x_to

        locs_from = (locs_from[:,0] * spatial_size[1] * spatial_size[2] +
                     locs_from[:,1] * spatial_size[2] +
                     locs_from[:,2]) * batch_size + locs_from[:,3]

        locs_to = (locs_to[:,0] * spatial_size[1] * spatial_size[2] +
                   locs_to[:,1] * spatial_size[2] +
                   locs_to[:,2]) * batch_size + locs_to[:,3]

        indicator_from = torch.zeros(
            spatial_size[0]*spatial_size[1]*spatial_size[2]*batch_size,
            dtype=torch.long, device=locs_from.device)

        indicator_to = indicator_from.clone()
        indicator_from[locs_from] = torch.arange(locs_from.shape[0], device=locs_from.device) + 1
        indicator_to[locs_to] = torch.arange(locs_to.shape[0], device=locs_to.device) + 1
        inds = torch.nonzero((indicator_from > 0) & (indicator_to > 0)).squeeze()

        feats_from = x_from[1].new_zeros(x_to[1].shape[0], x_from[1].shape[1])
        if inds.shape[0] > 0:
            feats_from[indicator_to[inds]-1] = x_from[1][indicator_from[inds]-1]
        x_to[1] = torch.cat([x_to[1], feats_from], 1)
        return x_to

    def update_sizes(self, input_max_dim, refine_max_dim):
        print('[model:update_sizes]', input_max_dim, refine_max_dim)
        if not isinstance(input_max_dim, (list, tuple, np.ndarray)):
            input_max_dim = [input_max_dim, input_max_dim, input_max_dim]
        if not isinstance(refine_max_dim, (list, tuple, np.ndarray)):
            refine_max_dim = [refine_max_dim, refine_max_dim, refine_max_dim]
        for k in range(3):
            self.encoder.process_sparse[0].p0.spatial_size[k] = torch.tensor( input_max_dim[k] )
            for h in range(len(self.shape_decoder)):
                self.shape_decoder[h].p0.spatial_size[k] = torch.tensor(refine_max_dim[k])
                refine_max_dim *= 2
                self.shape_decoder[h].n0.spatial_size[k] = torch.tensor(refine_max_dim[k])
            self.surfacepred.p0.spatial_size[k] = torch.tensor(refine_max_dim[k])

    def forward(self, x, loss_weights, locs_hier = None):

        outputs = []

        x = [x[0], torch.cat(x[1:], 1)]


        '''encode'''
        x, shape_out, feats_sparse = self.encoder(x)
        batch_size = x.shape[0]
        if self.use_skip_sparse:
            for k in range(len(feats_sparse)):
                '''convert SCN tensor to ([location, value], dim) tuple''' #print('[model] feats_sparse[%d]' % k, feats_sparse[k].spatial_size)
                feats_sparse[k] = ([feats_sparse[k].metadata.getSpatialLocations(feats_sparse[k].spatial_size), scn.OutputLayer(3)(feats_sparse[k])], feats_sparse[k].spatial_size)


        '''complete shape'''
        locs, feats, shape_out = self.dense_coarse_to_sparse(x, shape_out, truncation=3)
        outputs.append(shape_out)
        # for debug
        locs_hier_pred = [locs]
        x_sparse_shape = [locs, feats]
        for h in range(len(self.shape_decoder)):
            if loss_weights[h + 1] > 0:
                if self.use_skip_sparse:
                    x_sparse_shape = self.concat_skip(feats_sparse[len(self.shape_decoder) - h][0], x_sparse_shape,
                                                      feats_sparse[len(self.shape_decoder) - h][1], batch_size)
                x_sparse_shape, occ = self.shape_decoder[h](x_sparse_shape)
                locs_hier_pred.append (x_sparse_shape[0])
                outputs.append(occ)
            else:
                outputs.append([[], []])
        # surface prediction
        locs = x_sparse_shape[0]
        if loss_weights[-1] > 0:
            if self.use_skip_sparse:
                x_sparse_shape = self.concat_skip(feats_sparse[0][0], x_sparse_shape, feats_sparse[0][1], batch_size)
            x_sparse_shape = self.surfacepred(x_sparse_shape)
            shape_pred = ([locs, x_sparse_shape], outputs)
        else:
            shape_pred = ([[], []], outputs)


        '''complete motion with unkown regions'''
        if locs_hier is None: # if the GT geometry is not given, use predicted ones.
            if len(locs_hier_pred) == 4 :
                locs_hier = locs_hier_pred
            else :
                raise NotImplementedError()
        locs, feats = self.indexing_dense_to_sparse(x, locs_hier[0])
        x_sparse = [locs, feats]

        for h in range(len(self.mot_decoder)):
            x_sparse = self.concat_skip(
                feats_sparse[len(self.mot_decoder)-h][0], x_sparse,
                feats_sparse[len(self.mot_decoder)-h][1], batch_size)

            x_sparse = self.mot_decoder[h](x_sparse,
                                           locs_hier[h+1],
                                           feats_sparse[len(self.mot_decoder)-h-1][1], batch_size )

        x_sparse = self.concat_skip( feats_sparse[0][0], x_sparse,feats_sparse[0][1], batch_size)
        motion_pred = self.motion_pred(x_sparse)

        return  shape_pred, motion_pred



class MotionCompleteModel(nn.Module):
    ''' This model use SCN's built-in u-net'''
    def __init__(self, input_dim, input_nf):
        nn.Module.__init__(self)
        m = 16
        nPlanes = [ 16, 32, 64, 128 ]
        data_dim = 3  # spatial dimention
        reps = 1 # number of residual blecks
        self.sparseModel = scn.Sequential().add(
           scn.InputLayer(data_dim, input_dim, mode=3)).add(
           scn.SubmanifoldConvolution(data_dim, input_nf, m, 3, False)).add(
           scn.UNet(data_dim, reps, nPlanes, residual_blocks=True, downsample=[2,2])).add(
           scn.BatchNormReLU(m)).add(
           scn.OutputLayer(data_dim))
        self.mot_linear = nn.Linear(m, 3)

    def forward(self,x):
        x=self.sparseModel(x)
        trn = self.mot_linear(x)
        return trn


def euler2mat(angle):
    ''' Convert euler angles to rotation matrix.
    :param angle: rotation angle along 3 axis (in radians) -- size = [N 3]
    :return: Rotation matrix corresponding to the euler angles -- size = [N, 3,3]
    '''

    N,_ = angle.shape
    x, y, z = angle[:, 0:1 ], angle[:, 1:2 ], angle[:, 2: ]

    zeros = z * 0
    ones = zeros + 1

    cosz = torch.cos(z)
    sinz = torch.sin(z)
    zmat = torch.cat([cosz, -sinz, zeros,
                        sinz, cosz, zeros,
                        zeros, zeros, ones], dim=-1).view(-1, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)
    ymat = torch.cat([cosy, zeros, siny,
                        zeros, ones, zeros,
                        -siny, zeros, cosy], dim=-1).view(-1, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)
    xmat = torch.cat([ones, zeros, zeros,
                        zeros, cosx, -sinx,
                        zeros, sinx, cosx], dim=-1).view(-1, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat


class MotDecoder(nn.Module):
    def __init__(self, nf_in, nf, pass_occ, pass_feats, max_data_size, truncation=3):
        nn.Module.__init__(self)
        data_dim = 3
        self.pass_feats = pass_feats
        self.nf_in = nf_in
        self.nf = nf
        self.truncation = truncation
        self.p0 = scn.InputLayer(data_dim, max_data_size, mode=0)
        self.p1 = scn.SubmanifoldConvolution(data_dim, nf_in, nf, filter_size=FSIZE0, bias=False)
        self.p2 = scn.FullyConvolutionalNet(data_dim, reps=1, nPlanes=[nf, nf, nf], residual_blocks=True)
        self.p3 = scn.BatchNormReLU(nf * 3)
        self.p4 = scn.OutputLayer(data_dim)

        # upsampled
        self.n0 = scn.InputLayer(data_dim, max_data_size, mode=0)
        self.n1 = scn.SubmanifoldConvolution(data_dim, nf * 3, nf, filter_size=FSIZE0, bias=False)
        self.n2 = scn.BatchNormReLU(nf)
        self.n3 = scn.OutputLayer(data_dim)
        self.linear = nn.Linear(nf, 1)
        self.linearsdf = nn.Linear(nf, 1)
        self.linearmot = nn.Linear(nf, 3)

    def to_next_level_locs(self, locs, feats):  # upsample factor of 2 predictions
        assert (len(locs.shape) == 2)
        data_dim = locs.shape[-1] - 1  # assumes batch mode
        offsets = torch.nonzero(torch.ones(2, 2, 2)).long()  # 8 x 3
        locs_next = locs.unsqueeze(1).repeat(1, 8, 1)
        locs_next[:, :, :data_dim] *= 2
        locs_next[:, :, :data_dim] += offsets
        feats_next = feats.unsqueeze(1).repeat(1, 8, 1)
        return locs_next.view(-1, locs.shape[-1]), feats_next.view(-1, feats.shape[-1])

    def gather_feats_with_locs ( self, locs_from, feats_from, locs_to, dims, batch_size):

        nf = feats_from.shape[1]
        capacity = batch_size*dims[0]*dims[1]*dims[2]

        flatlocs_from = locs_from[:, 3] * dims[0] * dims[1] * dims[2] + \
                        locs_from[:, 0] * dims[1] * dims[2] + \
                        locs_from[:, 1] * dims[2] + \
                        locs_from[:, 2]
        flatlocs_to = locs_to[:, 3] * dims[0] * dims[1] * dims[2] + \
                      locs_to[:, 0] * dims[1] * dims[2] + \
                      locs_to[:, 1] * dims[2] + \
                      locs_to[:, 2]

        dense_feats = torch.zeros ( [ capacity, nf]).type_as(feats_from)
        dense_feats[flatlocs_from] = feats_from
        feats_to = dense_feats[flatlocs_to]

        return locs_to , feats_to

    def forward(self, x, next_level_locs, dim, batch_size ):

        input_locs = x[0]
        if len(input_locs) == 0:
            return [[], []], [[], []]

        x = self.p0(x)
        x = self.p1(x)
        x = self.p2(x)
        x = self.p3(x)
        x = self.p4(x)

        locs_unfilt, feats = self.to_next_level_locs(input_locs, x)

        next_level_locs, feats = self.gather_feats_with_locs ( locs_unfilt, feats, next_level_locs, dim, batch_size)

        x = self.n0([next_level_locs, feats])
        x = self.n1(x)
        x = self.n2(x)
        x = self.n3(x)

        return  [next_level_locs, x]

class MotionPrediction(nn.Module):
    def __init__(self, nf_in, nf,  max_data_size):
        nn.Module.__init__(self)
        data_dim = 3
        self.p0 = scn.InputLayer(data_dim, max_data_size, mode=0)
        self.p1 = scn.SubmanifoldConvolution(data_dim, nf_in, nf, filter_size=FSIZE0, bias=False)
        self.p2 = scn.FullyConvolutionalNet(data_dim, reps=1, nPlanes=[nf, nf, nf],
                                            residual_blocks=True)  # nPlanes=[nf, nf*2, nf*2], residual_blocks=True)
        self.p3 = scn.BatchNormReLU(nf * 3)
        self.p4 = scn.OutputLayer(data_dim)
        self.motlinear = nn.Linear(nf *3, 3)

    def forward(self, x):
        if len(x[0]) == 0:
            return [], []
        # x=self.sparseModel(x)
        # print('x(sparse)', x.shape)

        x = self.p0(x)
        x = self.p1(x)
        x = self.p2(x)
        x = self.p3(x)
        x = self.p4(x)

        trn = self.motlinear(x)
        return trn