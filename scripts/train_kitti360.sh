#!/bin/bash

#----------------------------------------------------------------------#
#                            INITIALIZATION                            #
#----------------------------------------------------------------------#
# This is the configuration we used for training our best DeepViewAgg
# model on KITTI-360 and achieve 58.3 mIoU on the held-out Test set,
# as stated in our paper: https://arxiv.org/abs/2204.07548

# Select you GPU
I_GPU=0

DATA_ROOT="/path/to/your/dataset/root/directory"                        # set your dataset root directory, where the data was/will be downloaded
EXP_NAME="My awesome KITTI-360 experiment"                              # whatever suits your needs
TASK="segmentation"
MODELS_CONFIG="${TASK}/multimodal/sparseconv3d"                         # family of multimodal models using the sparseconv3d backbone
MODEL_NAME="Res16UNet34-PointPyramid-early-cityscapes-interpolate"      # specific model name
DATASET_CONFIG="${TASK}/multimodal/kitti360-sparse"
TRAINING="kitti360_benchmark/sparseconv3d_rgb-pretrained-pyramid-0"     # training configuration for discriminative learning rate on the model
EPOCHS=60
CYLINDERS_PER_EPOCH=12000                                               # roughly speaking, 40 cylinders per window
TRAINVAL=False                                                          # True to train on Train+Val (eg before submission)
MINI=False                                                              # True to train on mini version of KITTI-360 (eg to debug)
BATCH_SIZE=4                                                            # 4 fits in a 32G V100. Can be increased at inference time, of course
WORKERS=4                                                               # adapt to your machine
BASE_LR=0.1                                                             # initial learning rate
LR_SCHEDULER='multi_step_kitti360' # learning rate scheduler for 60 epochs
EVAL_FREQUENCY=5                                                        # frequency at which metrics will be computed on Val. The less the faster the training but the less points on your validation curves
SUBMISSION=False                                                        # True if you want to generate files for a submission to the KITTI-360 3D semantic segmentation benchmark
CHECKPOINT_DIR=""                                                       # optional path to an already-existing checkpoint. If provided, the training will resume where it was left

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
data.sample_per_epoch=${CYLINDERS_PER_EPOCH} \
data.dataroot=${DATA_ROOT} \
data.train_is_trainval=${TRAINVAL} \
data.mini=${MINI} \
training.cuda=${I_GPU} \
training.batch_size=${BATCH_SIZE} \
training.epochs=${EPOCHS} \
training.num_workers=${WORKERS} \
training.optim.base_lr=${BASE_LR} \
training.wandb.log=True \
training.wandb.name=${EXP_NAME} \
tracker_options.make_submission=${SUBMISSION} \
training.checkpoint_dir=${CHECKPOINT_DIR}
