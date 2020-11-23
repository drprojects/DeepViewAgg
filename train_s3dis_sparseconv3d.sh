#!/bin/bash

# Launch training
for FOLD in 1 2 3 4 5 6
do
	python -W ignore train.py \
	task=segmentation \
	dataset=s3disfused-sparse \
	training=s3dis_benchmark/sparseconv3d \
	model_type=sparseconv3d \
	model_name=Res16UNet34 \
	lr_scheduler=multi_step_s3dis \
	wandb.log=True \
	training.cuda=0 \
	wandb.name=sparseconv3d-fold${FOLD} \
	data.fold=${FOLD}
done

# wandb API
# 7ca6580c36bd18b0d4e51b9c7dba6d8d147651d5