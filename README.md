###Code for data processing and training

#### Installation:  
This code was tested on Ubuntu 16.04, gcc 7.5, using PyTorch 1.3.1, Python 3.6.10, and CUDA 10.0, 

- Install [SparseConvNet]. refer to (https://github.com/facebookresearch/SparseConvNet).

- To visualize shape, you can install the marching cubes (from sgnn), or use other pakages.  
```
cd ./marching_cubes
python setup.py install
```



#### Prepare training chunks
- Download the volumetric data.  Then extract all sequence zip files of the volumetric data (see ./Data_API.md) to ```4DComplete/data/DeformingThings4D/raw```.
- Prepare the motion shape chunks for training. 
  Adjust the parameters (e.g. voxel size, crop size, number of hierarchical levels), refer to```4DComplete/torch/data_organization/MotionShapeCrop.py```  
```shell
python MotionShapeCrop.py
```
- get chunk list file .txt for train/val/test

#### Train Motion completion (structure known) 
The model is a simple Sparseconv UNet
```shell
python train.py 
    --mot True
    --data_path  ../data/DeformingThings4D/chunks/MotionShape_2cm_96_96_128 
    --train_file_list ../data/DeformingThings4D/train.txt
    --val_file_list ../data/DeformingThings4D/val.txt 
    --save ./snapshot
    --batch_size 8 
    --max_epoch 10 
    --gpu 2
```


#### Jointly train Motion and shape completion (structure unknown)
```shell
python train.py 
    --mot False
    --data_path  ../data/DeformingThings4D/chunks/MotionShape_2cm_96_96_128 
    --train_file_list ../data/DeformingThings4D/train.txt
    --val_file_list ../data/DeformingThings4D/val.txt 
    --save ./snapshot
    --batch_size 8 
    --max_epoch 10 
    --gpu 2
```
```shell
python train.py --mot False --data_path  ../data/DeformingThings4D/chunks/MotionShape_2cm_96_96_128 --train_file_list ../data/DeformingThings4D/train.txt --val_file_list ../data/DeformingThings4D/val.txt --save ./snapshot --batch_size 8 --max_epoch 10 --gpu 2
```
