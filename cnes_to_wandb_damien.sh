#!/bin/bash

# wandb API
# 7ca6580c36bd18b0d4e51b9c7dba6d8d147651d5


# Careful not to have any leading backslash for directory to be found
folder=scratch/wandb/segmentation/multimodal/s3disfused/3d_2d/sparse/no_pixel_height/5cm_1024x512-exact/fold5

# Mount remote directory in local filesystem
mkdir cnes_mount
sshfs cnes:$folder cnes_mount

# Recover and sync experiment directory
exp=____
cp --verbose -r cnes_mount/$exp/wandb/dry* ./$exp
wandb sync $exp

# Unmount remote directory in local filesystem
fusermount -u cnes_mount
rmdir cnes_mount
