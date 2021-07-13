#!/bin/bash

# wandb API
# 7ca6580c36bd18b0d4e51b9c7dba6d8d147651d5


#------------------------------------------------------------------------------#
#                                INITIALIZATION                                #
#------------------------------------------------------------------------------#

I_GPU=________

# DATA_ROOT='/media/drobert-admin/DATA2/datasets'  # IGN DATA2
# DATA_ROOT='/var/data/drobert/datasets'           # AI4GEO
# DATA_ROOT='/home/qt/robertda/scratch/datasets'   # CNES
# DATA_ROOT='/raid/dataset/pointcloud/data'        # ENGIE

# DATA_DIR=${DATA_ROOT}/s3dis/________
DATA_DIR=${DATA_ROOT}/s3dis/5cm_exact_1024x512

TASK=segmentation

# MODELS=${TASK}/________
MODELS=${TASK}/multimodal/no3d

# MODEL_NAME=_________
MODEL_NAME=RGB_ResNet18PPM_mean-feat

# DATASET=${TASK}/_________
DATASET=${TASK}/multimodal/s3disfused/no3d_5cm_1024x512-exact_no-pixel-height

# TRAINING=s3dis_benchmark/sparseconv3d
TRAINING=s3dis_benchmark/no3d_pretrained

# EXP_NAME=________
EXP_NAME=RGB_ResNet18PPM_mean-feat_LR-10-1-2-3_exact_1024x512

FOLD=5
EPOCHS=100
SPHERE_SAMPLES=2000  # 3000 for initial BATCH_SIZE=8
BATCH_SIZE=4
WORKERS=8
BASE_LR=0.1
LR_SCHEDULER=multi_step_s3dis
EVAL_FREQUENCY=5
export SPARSE_BACKEND=torchsparse
# export SPARSE_BACKEND=minkowski

# CHECKPOINT_DIR=""

#------------------------------------------------------------------------------#
#                                     RUN                                      #
#------------------------------------------------------------------------------#

python -W ignore train.py \
data=${DATASET} \
models=${MODELS} \
model_name=${MODEL_NAME} \
task=${TASK} \
training=${TRAINING} \
lr_scheduler=${LR_SCHEDULER} \
eval_frequency=${EVAL_FREQUENCY} \
data.fold=${FOLD} \
data.sample_per_epoch=${SPHERE_SAMPLES} \
data.dataroot=${DATA_DIR} \
training.cuda=${I_GPU} \
training.batch_size=${BATCH_SIZE} \
training.epochs=${EPOCHS} \
training.num_workers=${WORKERS} \
training.optim.base_lr=${BASE_LR} \
training.wandb.log=True \
training.wandb.name=${EXP_NAME}

# python -W ignore train.py \
# data=${DATASET} \
# models=${MODELS} \
# model_name=${MODEL_NAME} \
# task=${TASK} \
# training=${TRAINING} \
# lr_scheduler=${LR_SCHEDULER} \
# eval_frequency=${EVAL_FREQUENCY} \
# data.fold=${FOLD} \
# data.sample_per_epoch=${SPHERE_SAMPLES} \
# data.dataroot=${DATA_DIR} \
# training.cuda=${I_GPU} \
# training.batch_size=${BATCH_SIZE} \
# training.epochs=${EPOCHS} \
# training.num_workers=${WORKERS} \
# training.optim.base_lr=${BASE_LR} \
# training.wandb.log=True \
# training.wandb.name=${EXP_NAME}_resume \
# training.checkpoint_dir=${CHECKPOINT_DIR}
