from __future__ import division
from __future__ import print_function


import argparse
import os, sys, time
import torch
import numpy as np

import data_util
import model
import dataloader
import loss as loss_util
from utils import IOStream


parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--mot', type=bool, default=0, help='motion completion if true, else motion+shape completion')
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
parser.add_argument('--data_path', required=True, help='path to data')
parser.add_argument('--train_file_list', required=True, help='path to file list of train data')
parser.add_argument('--val_file_list', default='', help='path to file list of val data')
parser.add_argument('--save', default='./logs', help='folder to output model checkpoints')
# model params
parser.add_argument('--retrain', type=str, default='', help='model to load from')
parser.add_argument('--input_dim', type=int, default=0, help='voxel dim.')
parser.add_argument('--encoder_dim', type=int, default=8, help='pointnet feature dim')
parser.add_argument('--coarse_feat_dim', type=int, default=16, help='feature dim')
parser.add_argument('--refine_feat_dim', type=int, default=16, help='feature dim')
parser.add_argument('--no_pass_occ', dest='no_pass_occ', action='store_true')
parser.add_argument('--no_pass_feats', dest='no_pass_feats', action='store_true')
parser.add_argument('--use_skip_sparse', type=int, default=1, help='use skip connections between sparse convs')
parser.add_argument('--use_skip_dense', type=int, default=1, help='use skip connections between dense convs')
parser.add_argument('--no_logweight_target_sdf', dest='logweight_target_sdf', action='store_false')
# train params
parser.add_argument('--num_hierarchy_levels', type=int, default=4, help='#hierarchy levels (must be > 1).')
parser.add_argument('--num_iters_per_level', type=int, default=2000, help='#iters before fading in training for next level.')
parser.add_argument('--truncation', type=float, default=2, help='truncation in voxels')
parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
parser.add_argument('--max_epoch', type=int, default=5, help='number of epochs to train for')
parser.add_argument('--save_epoch', type=int, default=1, help='save every nth epoch')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
parser.add_argument('--decay_lr', type=int, default=100, help='decay learning rate by half every n epochs')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay.')
parser.add_argument('--weight_sdf_loss', type=float, default=1.0, help='weight sdf loss vs occ.')
# parser.add_argument('--weight_mot_loss', type=float, default=0.005, help='weight mot loss vs occ.')
parser.add_argument('--weight_missing_geo', type=float, default=5.0, help='weight missing geometry vs rest of sdf.')
parser.add_argument('--vis_dfs', type=int, default=0, help='use df (iso 1) to visualize')
parser.add_argument('--use_loss_masking', dest='use_loss_masking', action='store_true')
parser.add_argument('--no_loss_masking', dest='use_loss_masking', action='store_false')
parser.add_argument('--scheduler_step_size', type=int, default=0, help='#iters before scheduler step (0 for each epoch)')

parser.set_defaults(no_pass_occ=False, no_pass_feats=False, logweight_target_sdf=True, use_loss_masking=False)
args = parser.parse_args()
assert( not (args.no_pass_feats and args.no_pass_occ) )
assert( args.weight_missing_geo >= 1)
assert( args.num_hierarchy_levels > 1)

# args.input_dim = (64, 64, 128)
args.input_dim = (96, 96, 128)


args.decay_lr = args.max_epoch # done decay lr

# TODO delete this hardcode
args.mot = False

print(args)

# specify gpu
os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)


log_IO = IOStream(args.save)

MotionCompletion = args.mot

if MotionCompletion:
    """Motion Completion only, shape is known"""
    args.num_hierarchy_levels = 4
    args.input_nf = 3
    model = model.MotionCompleteModel( args.input_dim, args.input_nf ).cuda()

else :
    ''' Motion Shape Completion, shape is unkown'''
    args.num_hierarchy_levels = 4
    model = model.MotionShapeCompleteModel(
        args.input_dim,
        not args.no_pass_occ,
        not args.no_pass_feats,
        args.use_skip_sparse,
        args.use_skip_dense,
        log_IO,
        truncation=args.truncation).cuda()


optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
if args.retrain:
    print('loading model:', args.retrain)
    checkpoint = torch.load(args.retrain)
    args.start_epoch = args.start_epoch if args.start_epoch != 0 else checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict']) #, strict=False)
    optimizer.load_state_dict(checkpoint['optimizer'])
last_epoch = -1 if not args.retrain else args.start_epoch - 1
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_lr, gamma=0.5, last_epoch=last_epoch)


# data files
train_files, val_files = data_util.get_train_files(args.data_path, args.train_file_list, args.val_file_list)
_OVERFIT = False
if len(train_files) == 1:
    _OVERFIT = True
    args.use_loss_masking = False
print('#train files = ', len(train_files))
print('#val files = ', len(val_files))



if MotionCompletion: #DO MOTION COMPLETION'''
    train_dataset = dataloader.MotionCompleteDataset(train_files, args.input_dim, args.truncation, args.num_hierarchy_levels, 0)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=dataloader.collate_mocomplete)
else : #DO 4D COMPLETION'''
    train_dataset = dataloader.MotionShapeDataset(train_files, args.input_dim, args.truncation, args.num_hierarchy_levels, 0)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6, collate_fn=dataloader.collate_motionshape)



# if len(val_files) > 0:
has_val = False
if  has_val :
    val_dataset = dataloader.MotionShapeDataset(val_files, args.input_dim, args.truncation, args.num_hierarchy_levels, 0)
    print('val_dataset', len(val_dataset))
    val_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6, collate_fn=dataloader.collate_motionshape)


def bkup_experiment_files(log_dir):
    os.system('cp %s %s' % ("train.py", log_dir))  # bkp of model def
    os.system('cp %s %s' % ("dataloader.py", log_dir))  # bkp of train procedure
    os.system('cp %s %s' % ('model.py', log_dir))


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_loss_weights(iter, num_hierarchy_levels, num_iters_per_level, factor_l1_loss, log_IO):
    weights = np.zeros(num_hierarchy_levels+1, dtype=np.float32)
    cur_level = iter // num_iters_per_level
    if cur_level > num_hierarchy_levels:
        weights.fill(1)
        weights[-1] = factor_l1_loss
        if iter == (num_hierarchy_levels + 1) * num_iters_per_level:
            log_IO.write_to_terminal(f'[iter {iter}] updating loss weights: {weights}')
        return weights
    for level in range(0, cur_level+1):
        weights[level] = 1.0
    step_factor = 20
    fade_amount = max(1.0, min(100, num_iters_per_level//step_factor))
    fade_level = iter % num_iters_per_level
    cur_weight = 0.0
    l1_weight = 0.0
    if fade_level >= num_iters_per_level - fade_amount + step_factor:
        fade_level_step = (fade_level - num_iters_per_level + fade_amount) // step_factor
        cur_weight = float(fade_level_step) / float(fade_amount//step_factor)
    if cur_level+1 < num_hierarchy_levels:
        weights[cur_level+1] = cur_weight
    elif cur_level < num_hierarchy_levels:
        l1_weight = factor_l1_loss * cur_weight
    else:
        l1_weight = 1.0
    weights[-1] = l1_weight
    if iter % num_iters_per_level == 0 or (fade_level >= num_iters_per_level - fade_amount + step_factor and (fade_level - num_iters_per_level + fade_amount) % step_factor == 0):
        log_IO.write_to_terminal(f'[iter {iter}] updating loss weights: {weights}')
    return weights


def train_one_epoch_motion_complete(epoch, iter, dataloader, log_IO):

    model.train()
    start = time.time()

    if args.scheduler_step_size == 0:
        scheduler.step()

    for t, sample in enumerate(dataloader):


        if sample['bsize'] < args.batch_size:
            continue  # maintain same batch size for training

        inputs = sample['input']
        known = sample['known']
        motion = sample['motion']  # target motion

        # if args.input_nf == 4: # motion + mask as input
        #
        #     inputs[1] = torch.cat ( [ inputs[1], known.view(-1,1).float()] , -1 )
        # else:
        #     raise NotImplementedError()


        inputs[0] = inputs[0].cuda()
        inputs[1] = inputs[1].cuda()

        known = known.cuda()
        motion = motion.cuda()

        optimizer.zero_grad()

        pred_motion = model(inputs)

        # K: kown, UK: unkown
        pred_K = pred_motion[known]
        target_K = motion[known]
        pred_UK = pred_motion[~known]
        target_UK = motion[~known]
        K_rate = pred_K.shape[0] * 1.0 / pred_motion.shape[0] * 1.0

        '''l2 flow loss'''
        loss_K = loss_util.l2_flow_norm( pred_K, target_K )
        loss_UK = loss_util.l2_flow_norm( pred_UK, target_UK )
        '''end-point-error'''
        epe_K = loss_util.euclidean_distance(pred_K, target_K)
        epe_UK = loss_util.euclidean_distance(pred_UK, target_UK)
        '''cosine similarity'''
        cosim_K = loss_util.cosine_similarity(pred_K, target_K)
        cosim_UK = loss_util.cosine_similarity(pred_UK, target_UK)

        '''arap loss'''
        # _6nn = sample["6nn"].cuda()
        # known = known.float().view(-1,1)
        # mix_motion = inputs[1][:,:3] * known + pred_motion * (1-known)
        # arap_loss = loss_util.arap_loss( inputs[0],  mix_motion, rot, _6nn, args.input_dim)


        loss = K_rate * loss_K +  (1-K_rate)  * loss_UK + \
               K_rate * cosim_K + (1-K_rate)  * cosim_UK
               # arap_loss

        loss.backward()
        optimizer.step()

        iter += 1
        if args.scheduler_step_size > 0 and iter % args.scheduler_step_size == 0:
            scheduler.step()

        if iter % 20 == 0:
            took = time.time() - start
            iter_time = took / 20.
            start = time.time()
            display_dict = {"total_loss": loss.item(),
                            'loss_k':loss_K.item(),
                            'loss_uk': loss_UK.item(),
                            'epe_K' : epe_K.item(),
                            'epe_UK': epe_UK.item(),
                            'cosim_K': cosim_K.item(),
                            'cosim_UK': cosim_UK.item(),
                            # 'arap_loss': arap_loss.item(),
                            'k-rate': K_rate,
                            'learning_rate':get_lr(optimizer),
                            'time_per_iter':iter_time }


            log_IO.write_to_terminal('[iter %d/epoch %d]' % (iter, epoch) + str(display_dict))

        if iter % 6000 == 0:
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                       os.path.join(args.save, 'model-iter%s-epoch%s.pth' % (iter, epoch)))

    return iter, None


def train_one_epoch_motion_shape_complete(epoch, iter, dataloader, log_IO):


    display_step = 10
    display_dict = {}


    model.train()
    start = time.time()

    if args.scheduler_step_size == 0:
        scheduler.step()

    for t, sample in enumerate(dataloader):

        shape_loss_weights = get_loss_weights(iter, args.num_hierarchy_levels, args.num_iters_per_level, args.weight_sdf_loss, log_IO)

        # shape_loss_weights = [1.,1.,1.,1.,1.]

        if sample['bsize'] < args.batch_size:
            continue  # maintain same batch size for training

        inputs = sample['input']
        known = sample['known']
        motion = sample['motion']  # target motion
        sdfs = sample['sdf']  # target sdfs

        sdf_hierarchy = sample['sdf_hierarchy'] # target sdfs hier
        for h in range(len(sdf_hierarchy)):
            sdf_hierarchy[h] = sdf_hierarchy[h].cuda()

        target_locs_hier = sample['target_locs_hier']
        if shape_loss_weights[-1] == 1 :
            target_locs_hier = None

        inputs[0] = inputs[0].cuda()
        inputs[1] = inputs[1].cuda()
        inputs[2] = inputs[2].cuda()


        known = known.cuda()
        motion = motion.cuda()

        optimizer.zero_grad()


        pred_shape, pred_motion = model( inputs,  shape_loss_weights, target_locs_hier)

        loss = 0

        if pred_shape is not None :

            target_for_sdf, target_for_occs, target_sdf_hier = loss_util.compute_targets(
                sdfs.cuda(), [None] + sdf_hierarchy, args.num_hierarchy_levels, args.truncation, args.use_loss_masking,
                known)

            output_shape, shape_hier = pred_shape

            shape_loss, shape_loss_hier = loss_util.compute_shape_loss(
                output_shape, shape_hier,
                target_for_sdf, target_for_occs, target_sdf_hier,
                shape_loss_weights,
                args.truncation, args.logweight_target_sdf, args.weight_missing_geo, inputs[0], args.use_loss_masking, known)

            if iter%display_step == 0:
                shape_dict = {"shape_loss": shape_loss.item()}
                display_dict.update(shape_dict)


            loss += shape_loss


        if pred_motion is not None :


            pred_K = pred_motion[known]
            target_K = motion[known]
            pred_UK = pred_motion[~known]
            target_UK = motion[~known]
            K_rate = pred_K.shape[0] * 1.0 / pred_motion.shape[0] * 1.0

            '''l2 flow loss'''
            loss_K = loss_util.l2_flow_norm( pred_K, target_K )
            loss_UK = loss_util.l2_flow_norm( pred_UK, target_UK )
            '''end-point-error'''
            epe_K  = loss_util.euclidean_distance(pred_K, target_K)
            epe_UK = loss_util.euclidean_distance(pred_UK, target_UK)
            '''cosine similarity'''
            cosim_K = loss_util.cosine_similarity(pred_K, target_K)
            cosim_UK = loss_util.cosine_similarity(pred_UK, target_UK)

            motion_loss = K_rate * loss_K +  (1-K_rate)  * loss_UK + \
                   K_rate * cosim_K + (1-K_rate)  * cosim_UK

            if iter%display_step == 0:
                motion_dict = {"motion_loss": motion_loss.item(),
                                'loss_k':loss_K.item(),
                                'loss_uk': loss_UK.item(),
                                'epe_K' : epe_K.item(),
                                'epe_UK': epe_UK.item(),
                                'cosim_K': cosim_K.item(),
                                'cosim_UK': cosim_UK.item(),
                                # 'arap_loss': arap_loss.item(),
                                'k-rate': K_rate,
                                'learning_rate':get_lr(optimizer) }
                display_dict.update(motion_dict)


            loss += motion_loss


        loss.backward()
        optimizer.step()

        iter += 1
        if args.scheduler_step_size > 0 and iter % args.scheduler_step_size == 0:
            scheduler.step()

        if iter % display_step == 0:
            took = time.time() - start
            iter_time = took / display_step
            start = time.time()

            log_IO.write_to_terminal('[iter %d/epoch %d]' % (iter, epoch) + str(display_dict))

        if iter % 100 == 0:
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                       os.path.join(args.save, 'model-iter%s-epoch%s.pth' % (iter, epoch)))

    return iter, None


def test_one_epoch_motion_shape_complete(epoch, iter,  dataloader, log_IO):


    model.eval()


    num_batches = len(dataloader)

    EPE_K = 0
    EPE_UNK = 0

    L1_ALL = 0

    with torch.no_grad():

        for t, sample in enumerate(dataloader):

            if sample['bsize'] < args.batch_size:
                continue  # maintain same batch size for training

            inputs = sample['input']
            known = sample['known']
            motion = sample['motion']  # target motion
            sdfs = sample['sdf']  # target sdfs

            sdf_hierarchy = sample['sdf_hierarchy'] # target sdfs hier
            for h in range(len(sdf_hierarchy)):
                sdf_hierarchy[h] = sdf_hierarchy[h].cuda()

            target_locs_hier = sample['target_locs_hier']

            inputs[0] = inputs[0].cuda()
            inputs[1] = inputs[1].cuda()
            inputs[2] = inputs[2].cuda()


            known = known.cuda()
            motion = motion.cuda()

            pred_shape, pred_motion = model( inputs,  [1, 1, 1, 1, 1], target_locs_hier)


            #shape loss
            target_for_sdf, target_for_occs, target_sdf_hier = loss_util.compute_targets(
                sdfs.cuda(), [None] + sdf_hierarchy, args.num_hierarchy_levels, args.truncation, args.use_loss_masking,
                known)
            output_shape, shape_hier = pred_shape
            L1_surf = loss_util.compute_l1_predsurf_sparse_dense(
                output_shape[0], output_shape[1], target_for_sdf, None, False, False, None, batched=True)
            L1_ALL = L1_ALL + L1_surf


            #motion loss
            pred_K = pred_motion[known]
            target_K = motion[known]
            pred_UK = pred_motion[~known]
            target_UK = motion[~known]
            '''end-point-error'''
            epe_K = loss_util.euclidean_distance(pred_K, target_K)
            epe_UK = loss_util.euclidean_distance(pred_UK, target_UK)
            EPE_K = EPE_K + epe_K * args.batch_size
            EPE_UNK = EPE_UNK + epe_UK * args.batch_size



    L1_ALL = L1_ALL / num_batches
    EPE_K = EPE_K / num_batches
    EPE_UNK = EPE_UNK / num_batches

    display_dict = {
        'L1_surf_eval': L1_ALL.item() if torch.is_tensor(L1_ALL) else L1_ALL ,
        'epe_K_eval' :  EPE_K.item() if torch.is_tensor(EPE_K) else EPE_K,
        'epe_UK_eval':  EPE_UNK.item() if torch.is_tensor(EPE_UNK) else EPE_UNK}



    log_IO.write_to_terminal('validation results: [iter %d/epoch %d]' % (iter, epoch) + str(display_dict))


def main():
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    elif not _OVERFIT:
        print('warning: save dir %s exists' % args.save)



    log_IO.write_to_terminal(str(args).replace(",", "\n"))

    bkup_experiment_files(args.save)

    # start training
    log_IO.write_to_terminal('starting training...')

    iter = args.start_epoch * (len(train_dataset) // args.batch_size)

    for epoch in range(args.start_epoch, args.max_epoch):
        start = time.time()

        if MotionCompletion:
            iter, loss_weights = train_one_epoch_motion_complete(epoch, iter, train_dataloader, log_IO )
        else :
            iter, loss_weights = train_one_epoch_motion_shape_complete(epoch, iter, train_dataloader, log_IO )


        took = time.time() - start

        log_IO.write_to_terminal("epoch took time:" + str (took))

    log_IO.close()



if __name__ == '__main__':
    main()