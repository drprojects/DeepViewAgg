#!/bin/bash

# Recover input arguments
raw_dir=${1}
split=${2}

# Choose between the Train & Val and the Test data
if [ "$split" = "test" ]; then
    zip_file="data_3d_semantics_test.zip"
else
    zip_file="data_3d_semantics.zip"
fi

# # Create appropriate folders
# mkdir -p $raw_dir
cd $raw_dir 

# Download the data
if ! test -f "${zip_file}"; then
    wget --verbose "https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/6489aabd632d115c4280b978b2dcf72cb0142ad9/${zip_file}"
fi

# Unzip the data
echo "Unzipping ${zip_file} to ${raw_dir}..."
unzip -q -d ${raw_dir} ${zip_file}
# rm ${zip_file}

# Remove the train/ and test/ folder level so all sequences are in the same 
# directory 
if [ "$split" = "test" ]; then
    cd "${raw_dir}/data_3d_semantics/test"
    mv ./* ../
    rm -r "${raw_dir}/data_3d_semantics/test"
else
    cd "${raw_dir}/data_3d_semantics/train"
    mv ./* ../
    rm -r "${raw_dir}/data_3d_semantics/train"
fi