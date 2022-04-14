#!/bin/bash

# Recover input arguments
raw_dir=${1}
data_poses_dir=data_poses

# Create appropriate folders
mkdir -p ${raw_dir}
mkdir -p ${raw_dir}/${data_poses_dir}
cd $raw_dir 

# Download the data
zip_file="data_poses.zip"
if ! test -f "${zip_file}"; then
    wget --verbose "https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/89a6bae3c8a6f789e12de4807fc1e8fdcf182cf4/${zip_file}"
fi

# Unzip the data
echo "Unzipping ${zip_file} to ${data_poses_dir}..."
unzip -q -d ${data_poses_dir} ${zip_file}
# rm ${zip_file}
