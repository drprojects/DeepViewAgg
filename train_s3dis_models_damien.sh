#!/bin/bash

# wandb API
# 7ca6580c36bd18b0d4e51b9c7dba6d8d147651d5


#------------------------------------------------------------------------------#
#                                INITIALIZATION                                #
#------------------------------------------------------------------------------#

# source /opt/conda/etc/profile.d/conda.sh

I_GPU=1

CONDA_ENV=tp3d

PROJECT_DIR=~/projects/torch-points3d

#DATA_ROOT="/mnt/fa444ffd-fdb4-4701-88e7-f00297a8e29b/projects/datasets/s3dis_multimodal"  # DGX DATA
#DATA_ROOT="/media/drobert-admin/DATA/datasets/s3dis_tp3d_multimodal"  # IGN DATA
DATA_ROOT="/media/drobert-admin/DATA2/datasets/s3dis_tp3d_multimodal"  # IGN DATA2
#DATA_ROOT="/home/qt/robertda/scratch/datasets/s3dis/mm_s3disfused_5cm"  # CNES DATA
#DATA_ROOT="/var/data/drobert/datasets/s3dis/mm_s3disfused_5cm"  # AI4GEO

# DATA_ROOT=/home/qt/robertda/scratch/datasets/s3dis/s3disfused_5cm  # CNES DATA
#DATA_ROOT="/var/data/drobert/datasets/s3dis/s3disfused_5cm"  # AI4GEO

# DATA_ROOT="/raid/dataset/pointcloud/data/data/s3dis/2cm"  # ENGIE
# DATA_ROOT="/raid/dataset/pointcloud/data/data/s3dis/3cm"  # ENGIE
# DATA_ROOT="/raid/dataset/pointcloud/data/data/s3dis/4cm"  # ENGIE
# DATA_ROOT="/raid/dataset/pointcloud/data/data/s3dis/5cm"  # ENGIE
# DATA_ROOT="/raid/dataset/pointcloud/data/data/s3dis/3cm_256x128"  # ENGIE
# DATA_ROOT="/raid/dataset/pointcloud/data/data/s3dis/5cm_256x128"  # ENGIE
DATA_ROOT="/raid/dataset/pointcloud/data/data/s3dis/5cm_512x256"  # ENGIE

# MODEL_NAME=Res16UNet21-12
# MODEL_NAME=Res16UNet21-15
# MODEL_NAME=Res16UNet21-15-large
# MODEL_NAME=Res16UNet13-12
# MODEL_NAME=Res16UNet13-15

# MODEL_NAME=Pure2D_Res16UNet21-12
# MODEL_NAME=Pure2D_Res16UNet21-15
# MODEL_NAME=Pure2D_Res16UNet13-12
# MODEL_NAME=Pure2D_Res16UNet13-15
# MODEL_NAME=Pure2D_Res16UNet21-12_GN_WS
# MODEL_NAME=Pure2D_Res16UNet21-15_GN_WS-k4
# MODEL_NAME=Pure2D_Res16UNet21-15_GN_WS-k2
# MODEL_NAME=Pure2D_Res16UNet21-15_GN_WS-k4-large
MODEL_NAME=Pure2D_Res16UNet21-15_GN_WS-k2-large
# MODEL_NAME=Pure2D_Res16UNet13-12_GN_WS
# MODEL_NAME=Pure2D_Res16UNet13-15_GN_WS

MODEL_TYPE=no3d
# MODEL_TYPE=sparseconv3d

export SPARSE_BACKEND=torchsparse
# export SPARSE_BACKEND=minkowski

TASK=segmentation/multimodal
# TASK=segmentation

# DATASET=s3disfused
DATASET=s3disfused-no3d
# DATASET=s3disfused-no3d-256x128
# DATASET=s3disfused-no3d-512x256
# DATASET=s3disfused-sparse
# DATASET=s3disfused-sparse-norgb

VOXEL=0.05
# VOXEL=0.04
# VOXEL=0.03
# VOXEL=0.02

TRAINING=s3dis_benchmark/sparseconv3d

EPOCHS=200

WORKERS=4

# BATCH_SIZE=8
BATCH_SIZE=4

BASE_LR=0.01
# BASE_LR=0.1

# LR_SCHEDULER=multi_step_s3dis
LR_SCHEDULER=multi_step_s3dis_image

#------------------------------------------------------------------------------#


FOLD=5

EXP_NAME=${MODEL_NAME}_${DATASET}_fold${FOLD}

python -W ignore train.py \
task=${TASK} \
dataset=${DATASET} \
training=${TRAINING} \
model_type=${MODEL_TYPE} \
model_name=${MODEL_NAME} \
lr_scheduler=${LR_SCHEDULER} \
data.fold=${FOLD} \
data.first_subsampling=${VOXEL} \
data.dataroot=${DATA_ROOT} \
wandb.log=True \
wandb.name=${EXP_NAME} \
training.cuda=${I_GPU} \
training.batch_size=${BATCH_SIZE} \
training.epochs=${EPOCHS} \
training.num_workers=${WORKERS} \
training.optim.base_lr=${BASE_LR}

