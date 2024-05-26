import matplotlib.pyplot as plt
import re
import os, sys, time, shutil

sys.path.append("..")

from model import MotionShapeCompleteModel
import data_util
from dataloader import MotionShapeDataset, collate_motionshape
import torch
from utils import IOStream
from loss import compute_shape_loss, compute_targets


def plot_overfit_curve():
    run_log = open('run.log', 'r')
    lines_ = run_log.readlines()
    
    count = 0
    # Strips the newline character
    x_y = dict()
    for line in lines_:
        if "'shape_loss'" in line:
            line = line.strip()

            iter, epoch, shape_loss = re.findall(r"^\[iter ([0-9]+)\/epoch ([0-9]+)\]\{'shape_loss': ([0-9]+\.[0-9]+)\}$", line)[0]
            x_y[int(iter)] = float(shape_loss)

    x_y = sorted(x_y.items()) # sorted by key, return a list of tuples

    x, y = zip(*x_y)

    with plt.style.context('ggplot'):
        plt.plot(x, y)
        plt.xlabel('Epochs')
        plt.ylabel('Shape loss')
        plt.title('Training curve (overfitting)')
        # plt.show()
        plt.savefig('Training curve (overfitting).png')

    plt.clf()
    i = 1
    x_list, y_list = [], []
    for x_i, y_i in zip(x, y):
        if x_i%200 == 0 and x_i != 0:
            with plt.style.context('ggplot'):
                plt.plot(x_list, y_list)
                plt.xlabel('Epochs')
                plt.ylabel('Shape loss')
                plt.title(f'Training curve (overfitting) (level {i})')
                plt.savefig(f'Training curve (overfitting) {i}.png')
            plt.clf()
            i+=1
            x_list, y_list = [x_i], [y_i]
        else:
            x_list.append(x_i)
            y_list.append(y_i)


def create_ply(dims, locs, output_ply="/home/adl4cv/projects/sdfs28.ply"):

    max_height = 100
    up_axis = 0

    dimz, dimy, dimx = dims

    if dimz > max_height :
        mask_input = locs[:, up_axis] < max_height
        mask_input = mask_input * (locs[:, 1] < max_height )
        mask_input = mask_input * (locs[:, 2] < max_height)

        locs  = locs[mask_input]

    data_util.visualize_sparse_sdf_as_cubes(locs, output_ply )


def test_visualize(retrain='/home/adl4cv/projects/adl4cv/torch/snapshot_run_1/model-iter1000-epoch1000.pth',
                   data_path = '../../data/DeformingThings4D/chunks/MotionShape_2cm_96_96_128',
                   train_file_list = '../../data/DeformingThings4D/train.txt',
                   val_file_list = '../../data/DeformingThings4D/val.txt'):

    num_hierarchy_levels = 4
    batch_size=1
    input_dim = (96, 96, 128)
    no_pass_occ=False
    no_pass_feats=False
    logweight_target_sdf=True
    use_loss_masking=False
    use_skip_sparse=1
    use_skip_dense=1
    truncation=2
    weight_missing_geo=5.0
    log_IO = IOStream('', is_training=False)

    model = MotionShapeCompleteModel(
        input_dim,
        not no_pass_occ,
        not no_pass_feats,
        use_skip_sparse,
        use_skip_dense,
        log_IO,
        truncation=truncation).cuda()

    plots_dir = './ply_vis/'
    if not os.path.isdir(plots_dir):
        print('Create ', plots_dir, ' folder')
        os.mkdir(plots_dir)

    print('loading model:', retrain)
    checkpoint = torch.load(retrain)
    model.load_state_dict(checkpoint['state_dict']) #, strict=False)


    # data files
    train_files, val_files = data_util.get_train_files(data_path, train_file_list, val_file_list)
    print('#train files = ', len(train_files))
    print('#val files = ', len(val_files))

    train_dataset = MotionShapeDataset(train_files, input_dim, truncation, num_hierarchy_levels, 0)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6, collate_fn=collate_motionshape)

    model.eval()
    start = time.time()

    for t, sample in enumerate(dataloader):

        shape_loss_weights = [1.,1.,1.,1.,1.]

        if sample['bsize'] < batch_size:
            continue  # maintain same batch size for training

        inputs = sample['input']
        known = sample['known']
        sdfs = sample['sdf']  # target sdfs

        if not os.path.isdir(os.path.join(plots_dir, sample['name'][0])):
            os.mkdir(os.path.join(plots_dir, sample['name'][0]))
        else:
            shutil.rmtree(os.path.join(plots_dir, sample['name'][0]))
            os.mkdir(os.path.join(plots_dir, sample['name'][0]))

        create_ply(list(sdfs.shape[2:]), inputs[0][:, :3], output_ply=os.path.join(plots_dir, sample['name'][0], "input.ply"))

        sdf_hierarchy = sample['sdf_hierarchy'] # target sdfs hier
        for h in range(len(sdf_hierarchy)):
            sdf_hierarchy[h] = sdf_hierarchy[h].cuda()

        # target_locs_hier = sample['target_locs_hier']
        # if shape_loss_weights[-1] == 1 :
            # target_locs_hier = None
        target_locs_hier = None

        inputs[0] = inputs[0].cuda()
        inputs[1] = inputs[1].cuda()
        inputs[2] = inputs[2].cuda()


        known = known.cuda()

        pred_shape, pred_motion = model( inputs,  shape_loss_weights, target_locs_hier)

        loss = 0

        if pred_shape is not None :

            target_for_sdf, target_for_occs, target_sdf_hier = compute_targets(
                sdfs.cuda(), [None] + sdf_hierarchy, num_hierarchy_levels, truncation, use_loss_masking,
                known)

            output_shape, shape_hier = pred_shape

            shape_loss, shape_loss_hier = compute_shape_loss(
                output_shape, shape_hier,
                target_for_sdf, target_for_occs, target_sdf_hier,
                shape_loss_weights,
                truncation, logweight_target_sdf, weight_missing_geo, inputs[0], use_loss_masking, known)

            # plt create for prediction
            output_surf_locs, output_surf_field = output_shape
            create_ply(list(sdfs.shape[2:]), output_surf_locs[:, :3], output_ply=os.path.join(plots_dir, sample['name'][0], "prediction.ply"))
            print("shape_loss: ", shape_loss.item())

        took = time.time() - start



if __name__ == '__main__':
    test_visualize()