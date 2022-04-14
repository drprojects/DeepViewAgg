#!/bin/bash

# Recover input arguments
raw_dir=${1}

# Create appropriate folders
mkdir -p $raw_dir
cd $raw_dir

# Download the data
zip_file="calibration.zip"
if ! test -f "${zip_file}"; then
    wget --verbose "https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/384509ed5413ccc81328cf8c55cc6af078b8c444/${zip_file}"
fi

# Unzip the data
echo "Unzipping the data..."
unzip -q -d ${raw_dir} ${zip_file}
# rm ${zip_file}
