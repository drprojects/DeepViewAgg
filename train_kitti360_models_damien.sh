#!/bin/bash

# wandb API
# 7ca6580c36bd18b0d4e51b9c7dba6d8d147651d5


# ------------------------------------------------------------------------------#
#                                INITIALIZATION                                #
# ------------------------------------------------------------------------------#

I_GPU=________

# DATA_ROOT='/media/drobert-admin/DATA2/datasets'  # IGN DATA2
# DATA_ROOT='/var/data/drobert/datasets'           # AI4GEO
# DATA_ROOT='/home/qt/robertda/scratch/datasets'   # CNES
# DATA_ROOT='/raid/dataset/pointcloud/data'        # ENGIE

# DATA_DIR=${DATA_ROOT}/kitti360/________
DATA_DIR=${DATA_ROOT}/kitti360/5cm

TASK=segmentation

# MODELS=${TASK}/________
# MODELS=${TASK}/multimodal/no3d
MODELS=${TASK}/sparseconv3d

# MODEL_NAME=_________
MODEL_NAME=Res16UNet34

# DATASET=${TASK}/_________
DATASET=${TASK}/kitti360-sparse

TRAINING=kitti360_benchmark/sparseconv3d
# TRAINING=kitti360_benchmark/sparseconv3d_rgb-pretrained-0
# TRAINING=kitti360_benchmark/no3d_pretrained

# EXP_NAME=________
EXP_NAME=${MODEL_NAME}

EPOCHS=60
CYLINDERS_PER_EPOCH=12000  # Roughly speaking, 40 cylinders per window
# BATCH_SIZE=4
BATCH_SIZE=8
WORKERS=4
BASE_LR=0.1
LR_SCHEDULER=multi_step_kitti360_${EPOCHS}
EVAL_FREQUENCY=1
export SPARSE_BACKEND=torchsparse
# export SPARSE_BACKEND=minkowski
SUBMISSION=False
#SUBMISSION=True

# CHECKPOINT_DIR=""

# ------------------------------------------------------------------------------#
#                                     RUN                                      #
# ------------------------------------------------------------------------------#

python -W ignore train.py \
data=${DATASET} \
models=${MODELS} \
model_name=${MODEL_NAME} \
task=${TASK} \
training=${TRAINING} \
lr_scheduler=${LR_SCHEDULER} \
eval_frequency=${EVAL_FREQUENCY} \
data.sample_per_epoch=${CYLINDERS_PER_EPOCH} \
data.dataroot=${DATA_DIR} \
training.cuda=${I_GPU} \
training.batch_size=${BATCH_SIZE} \
training.epochs=${EPOCHS} \
training.num_workers=${WORKERS} \
training.optim.base_lr=${BASE_LR} \
training.wandb.log=True \
training.wandb.name=${EXP_NAME} \
tracker_options.make_submission=${SUBMISSION}
