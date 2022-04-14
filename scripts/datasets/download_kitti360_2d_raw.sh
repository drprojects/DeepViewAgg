#!/bin/bash

# Supported sequences and cameras
sequences=(
    "2013_05_28_drive_0000_sync"
    "2013_05_28_drive_0002_sync"
    "2013_05_28_drive_0003_sync"
    "2013_05_28_drive_0004_sync"
    "2013_05_28_drive_0005_sync"
    "2013_05_28_drive_0006_sync"
    "2013_05_28_drive_0007_sync"
    "2013_05_28_drive_0008_sync"
    "2013_05_28_drive_0009_sync"
    "2013_05_28_drive_0010_sync"
    "2013_05_28_drive_0018_sync")
cameras=("0" "1" "2" "3")

# Recover input arguments
raw_dir=${1}
data_2d_dir=data_2d_raw
sequence=${2}
camera=${3}

# Make sure the provided sequence exists
if [[ ! " ${sequences[*]} " =~ " ${sequence} " ]]
then
    echo "Unknown sequence '${sequence}'. Supported sequences are: "`echo ${sequences[*]}`
    exit 1
fi

# Make sure the provided camera exists
if [[ ! " ${cameras[*]} " =~ " ${camera} " ]]
then
    echo "Unknown camera '${camera}'. Supported cameras are: "`echo ${cameras[*]}`
    exit 1
fi

# Create appropriate folders
mkdir -p $raw_dir
mkdir -p $raw_dir/$data_2d_dir
cd $raw_dir 

# Download the data
zip_file=${sequence}_image_0${camera}.zip
if ! test -f "${zip_file}"; then
    wget --verbose "https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/data_2d_raw/${zip_file}"
fi

# Unzip the data
echo "Unzipping the data..."
unzip -q -d ${data_2d_dir} ${zip_file}
# rm ${zip_file}
