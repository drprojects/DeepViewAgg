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

#RESOLUTION_3D_CM=5
#HEIGHT=128
#VOXEL=$(LANG=C printf %.3f\\n "$(( 10**3 * ${RESOLUTION_3D_CM} / 100  ))e-3")
#let WIDTH=HEIGHT*2

VOXEL=0.05
# VOXEL=0.04
# VOXEL=0.03

MACHINE_DATA_ROOT='/media/drobert-admin/DATA2/datasets/s3dis'  # IGN DATA2
#MACHINE_DATA_ROOT='/var/data/drobert/datasets/s3dis'          # AI4GEO
#MACHINE_DATA_ROOT='/home/qt/robertda/scratch/datasets/s3dis'  # CNES
#MACHINE_DATA_ROOT='/raid/dataset/pointcloud/data/s3dis'      # ENGIE

#DATA_ROOT=${MACHINE_DATA_ROOT}/2cm
#DATA_ROOT=${MACHINE_DATA_ROOT}/3cm
DATA_ROOT=${MACHINE_DATA_ROOT}/5cm
#DATA_ROOT=${MACHINE_DATA_ROOT}/5cm_256x128
#DATA_ROOT=${MACHINE_DATA_ROOT}/5cm_512x256

### XYZRGB and XYZ
MODEL_NAME=Res16UNet21-15

### RGB
#MODEL_NAME=Res16UNet21-15_GN_WS
#MODEL_NAME=Res16UNet21-15_GN
#MODEL_NAME=Res16UNet21-15_BN

#MODEL_TYPE=no3d
MODEL_TYPE=sparseconv3d

export SPARSE_BACKEND=torchsparse
# export SPARSE_BACKEND=minkowski

# TASK=segmentation/multimodal
TASK=segmentation

#DATASET=s3disfused
#DATASET=s3disfused-no3d
DATASET=s3disfused-sparse
#DATASET=s3disfused-sparse-norgb

TRAINING=s3dis_benchmark/sparseconv3d

CHECKPOINT_DIR=""

EPOCHS=200

WORKERS=2

BATCH_SIZE=8
#BATCH_SIZE=4

BASE_LR=0.1

LR_SCHEDULER=multi_step_s3dis

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
