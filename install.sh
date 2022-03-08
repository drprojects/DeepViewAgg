#!/bin/bash

# Recover the project directory from the position of the install.sh script
HERE=`dirname $0`
HERE=`realpath $HERE`


# Local variables
PROJECT_NAME=deep_view_aggregation
YML_FILE=${HERE}/${PROJECT_NAME}.yml
TORCH=1.7.0


# Installation script for Anaconda3 environments
echo "#############################################"
echo "#                                           #" 
echo "#           Deep View Aggregation           #"
echo "#                 Installer                 #"
echo "#                                           #" 
echo "#############################################"
echo
echo


echo "_______________ Prerequisites _______________"
echo "  - conda"
echo "  - cuda >= 10.2 (tested with 10.2, 11.2, 11.4)"
echo "  - gcc >= 7"
echo
echo


echo "____________ Pick conda install _____________"
echo
# Recover the path to conda on your machine
CONDA_DIR=`realpath ~/anaconda3`
CONDA_DIR=/home/ign.fr/drobert/anaconda3 ###############"""**************************-------------------------////////////////////////////////

while (test -z $CONDA_DIR) || [ ! -d $CONDA_DIR ]
do
    echo "Could not find conda at: "$CONDA_DIR
    read -p "Please provide you conda install directory: " CONDA_DIR
    CONDA_DIR=`realpath $CONDA_DIR`
done

echo "Using conda conda found at: ${CONDA_DIR}/etc/profile.d/conda.sh"
source ${CONDA_DIR}/etc/profile.d/conda.sh
echo
echo


echo "_____________ Pick CUDA version _____________"
echo

CUDA_SUPPORTED=(10.2 11.2 11.4)
CUDA_VERSION=`nvcc --version | grep release | sed 's/.* release //' | sed 's/, .*//'`

# If CUDA version not supported, ask whether to proceed
if [[ ! " ${CUDA_SUPPORTED[*]} " =~ " ${CUDA_VERSION} " ]]
then
    echo "Found CUDA ${CUDA_VERSION} installed, is not among tested versions: "`echo ${CUDA_SUPPORTED[*]}`
    read -p "This may cause downstream errors when installing dependencies. Do you want to proceed anyways ? [y/n] " -n 1 -r; echo
    if !(test -z $REPLY) && [[ ! $REPLY =~ ^[Yy]$ ]]
    then
        exit 1
    fi
fi

echo "Chosen CUDA version: ${CUDA_VERSION}"
cuXXX=`echo "cu"${CUDA_VERSION} | sed 's/\.//'`
echo
echo


echo "________________ Installation _______________"
echo

# Create deep_view_aggregation environment from yml
conda env create -f ${YML_FILE}

# Activate the env
source ${CONDA_DIR}/etc/profile.d/conda.sh  
conda activate ${PROJECT_NAME}

# Dependencies not installed from the .yml
pip install torch==1.7.0 torchvision==0.8.0 --no-cache-dir 
pip install torch-points-kernels
pip install torchnet

# Install torch-geometric and dependencies
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${cuXXX}.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${cuXXX}.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${cuXXX}.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${cuXXX}.html
pip install torch-geometric

# torch-points3d dependencies
pip install omegaconf
pip install wandb
pip install tensorboard
pip install plyfile
pip install hydra-core==1.1.0
pip install pytorch-metric-learning

# Install MinkowskiEngine
sudo apt install libopenblas-dev
pip install -U MinkowskiEngine==v0.4.3 --install-option="--blas=openblas" -v --no-deps

# Install torchsparse
sudo apt-get install libsparsehash-dev
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git

# Install plotly for jupyter
conda install jupyterlab "ipywidgets=7.5"
jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget@4.14.1

# Install FAISS
conda install -c conda-forge faiss-gpu=1.6.5

### Make deep_view_aggregation kernel accessible on jupyterlab
