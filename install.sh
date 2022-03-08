#!/bin/bash

# Recover the project directory from the position of the install.sh script
PROJECT_DIR=`dirname $0`
PROJECT_DIR=`realpath $PROJECT_DIR`


# Local variables
YML_FILE=${PROJECT_DIR}/deep_view_aggregation.yml
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

# Accept with y / Y / ENTER
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
# List installed CUDA versions
CUDA_VERSIONS_LIST=(`ls /usr/local/ | grep ^cuda- | sed 's/cuda-//'`)

if [[ ${#CUDA_VERSIONS_LIST[@]} == 0 ]]; then
    echo "Could not find any CUDA install in '/usr/local/'. Make sure you have CUDA installed there first (tested versions are 10.2, 11.2 and 11.4)."
    exit 1
fi

if [[ ${#CUDA_VERSIONS_LIST[@]} == 1 ]]; then
    CUDA_VERSION=${CUDA_VERSIONS_LIST[0]}
else
    echo "Found "`echo ${#CUDA_VERSIONS_LIST[@]}`" CUDA versions installed: "`echo "${CUDA_VERSIONS_LIST[*]}"`
    read -p "Choose a CUDA version for the environment: " CUDA_VERSION
    echo
fi

while [[ ! " ${CUDA_VERSIONS_LIST[*]} " =~ " ${CUDA_VERSION} " && ${#CUDA_VERSIONS_LIST[@]} > 2 ]]
do
    echo "'$CUDA_VERSION' not among supported CUDA versions: "$CUDA_VERSIONS_LIST
    read -p "Choose a CUDA version for the environment: " CUDA_VERSION
    echo
done

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
conda activate tp3d_dev

# Dependencies not installed from the .yml
pip install torch==1.7.0 torchvision==0.8.0 --no-cache-dir 
pip install torch-points-kernels
pip install torchnet

# Install torch-geometric and dependencies
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
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
