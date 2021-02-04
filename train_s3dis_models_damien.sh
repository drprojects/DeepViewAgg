#!/bin/bash

# wandb API
# 7ca6580c36bd18b0d4e51b9c7dba6d8d147651d5

########################################################################
# Sparseconv3d baseline on S3DIS
MODEL_NAME=ResUNet34_0_2_2_2_2

for FOLD in 1 2 3 4 5 6
do
	python -W ignore train.py \
	task=segmentation \
	dataset=s3disfused-sparse \
	training=s3dis_benchmark/sparseconv3d \
	model_type=sparseconv3d \
	model_name=${MODEL_NAME} \
	lr_scheduler=multi_step_s3dis \
	wandb.log=True \
	training.cuda=0 \
	wandb.name=${MODEL_NAME}-fold${FOLD} \
	data.fold=${FOLD}
done

########################################################################
# Pure2D baseline on S3DIS
#MODEL_NAME=Pure2D_ResUNet34_0_2_2_2_2
MODEL_NAME=Pure2D_ResUNet34_0_1_1_1_1

for FOLD in 5
do
	python -W ignore train.py \
	task=segmentation/multimodal \
	dataset=s3disfused-no3d \
	training=s3dis_benchmark/sparseconv3d \
	model_type=no3d \
	model_name=${MODEL_NAME} \
	lr_scheduler=multi_step_s3dis \
	wandb.log=True \
	training.cuda=0 \
	wandb.name=${MODEL_NAME}-fold${FOLD} \
	data.fold=${FOLD}
done


