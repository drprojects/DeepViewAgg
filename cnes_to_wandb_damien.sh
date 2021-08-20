#!/bin/bash

# wandb API
# 7ca6580c36bd18b0d4e51b9c7dba6d8d147651d5

# Careful not to have any leading backslash for directory to be found
folder=scratch/wandb/segmentation/multimodal/s3disfused/3d_2d/sparse/no_pixel_height/5cm_1024x512-exact/fold5
exp=____

# Print results even when not finished
cat ~/$folder/$exp/wandb/dry*/output.log | grep "\(test_miou =\)\|\(val_miou =\)\|\(EPOCH \)"
tail -30 ~/$folder/$exp/wandb/dry*/output.log

# Mount remote directory in local filesystem
mkdir cnes_mount
sshfs cnes:$folder cnes_mount

# Recover and sync experiment directory
declare -a explist=( "exp1" "exp2" )
for exp in ${explist[@]}; do
    echo
    echo --------------------------------
    echo $exp
    echo --------------------------------
    cp --verbose -r cnes_mount/$exp/wandb/dry* ./$exp
  wandb sync $exp
done

# Unmount remote directory in local filesystem
fusermount -u cnes_mount
rmdir cnes_mount
