#!/bin/bash

#----------------------------------------------------------------------#
#                            INITIALIZATION                            #
#----------------------------------------------------------------------#
# This is the configuration we used for training our best DeepViewAgg
# model on S3DIS Fold 5. If you want to achieve 74.7 mIoU on S3DIS
# 6-Fold, as stated in our paper https://arxiv.org/abs/2204.07548,
# you will need to run this experiment on each of the 6 folds.

# Select you GPU
I_GPU=0

DATA_ROOT="/path/to/your/dataset/root/directory"                        # set your dataset root directory, where the data was/will be downloaded
EXP_NAME="My_awesome_S3DIS_experiment"                                  # whatever suits your needs
TASK="segmentation"
MODELS_CONFIG="${TASK}/multimodal/sparseconv3d"                         # family of multimodal models using the sparseconv3d backbone
MODEL_NAME="Res16UNet34-L4-early-ade20k-interpolate"                    # specific model name
DATASET_CONFIG="${TASK}/multimodal/s3disfused-sparse"
TRAINING="s3dis_benchmark/sparseconv3d_rgb-pretrained-0"                # training configuration for discriminative learning rate on the model
FOLD=5                                                                  # S3DIS Fold that will be used as Test set
EPOCHS=200
SPHERE_SAMPLES=2000                                                     # number of spherical samples per training epoch
BATCH_SIZE=4                                                            # 4 fits in a 32G V100. Can be increased at inference time, of course
WORKERS=4                                                               # adapt to your machine
BASE_LR=0.1                                                             # initial learning rate
LR_SCHEDULER='multi_step'                                               # learning rate scheduler for 60 epochs
EVAL_FREQUENCY=5                                                        # frequency at which metrics will be computed on Val. The less the faster the training but the less points on your validation curves
CHECKPOINT_DIR=''                                                       # optional path to an already-existing checkpoint. If provided, the training will resume where it was left

export SPARSE_BACKEND=torchsparse
# export SPARSE_BACKEND=minkowski

#----------------------------------------------------------------------#
#                                 RUN                                  #
#----------------------------------------------------------------------#

python -W ignore train.py \
data=${DATASET_CONFIG} \
models=${MODELS_CONFIG} \
model_name=${MODEL_NAME} \
task=${TASK} \
training=${TRAINING} \
lr_scheduler=${LR_SCHEDULER} \
eval_frequency=${EVAL_FREQUENCY} \
data.fold=${FOLD} \
data.sample_per_epoch=${SPHERE_SAMPLES} \
data.dataroot=${DATA_ROOT} \
training.cuda=${I_GPU} \
training.batch_size=${BATCH_SIZE} \
training.epochs=${EPOCHS} \
training.num_workers=${WORKERS} \
training.optim.base_lr=${BASE_LR} \
training.wandb.log=True \
training.wandb.name=${EXP_NAME} \
training.checkpoint_dir=${CHECKPOINT_DIR}
