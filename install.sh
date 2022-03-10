#!/bin/bash

# Recover the project directory from the position of the install.sh script
HERE=`dirname $0`
HERE=`realpath $HERE`


# Local variables
PROJECT_NAME=deep_view_aggregation
YML_FILE=${HERE}/${PROJECT_NAME}.yml
#TORCH=1.7.0 ############################************************!!!!!!!!!!!!!!!!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
TORCH=1.7.1
CUDA_SUPPORTED=(10.1 10.2 11.0 11.2 11.4)


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
echo "  - cuda >= 10.1 (tested with `echo ${CUDA_SUPPORTED[*]}`)"
echo "  - gcc >= 7"
echo
echo


echo "____________ Pick conda install _____________"
echo
# Recover the path to conda on your machine
CONDA_DIR=`realpath ~/anaconda3`

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
CUDA_MAJOR=`echo $CUDA_VERSION | sed 's/\..*//'`
CUDA_MINOR=`echo $CUDA_VERSION | sed 's/.*\.//'`
CUDA_MINOR_TORCH=`echo $CUDA_MINOR`
CUDA_MINOR_TV=`echo $CUDA_MINOR`
CUDA_MINOR_PYG=`echo $CUDA_MINOR`

# If CUDA version not supported, ask whether to proceed
if [[ ! " ${CUDA_SUPPORTED[*]} " =~ " ${CUDA_VERSION} " ]]
then
    echo "Found CUDA ${CUDA_VERSION} installed, is not among tested versions: "`echo ${CUDA_SUPPORTED[*]}`
    echo "This may cause downstream errors when installing PyTorch and PyTorch Geometric dependencies, which you might solve by manually modifying setting the wheels in this script."
    read -p "Proceed anyways ? [y/n] " -n 1 -r; echo
    if !(test -z $REPLY) && [[ ! $REPLY =~ ^[Yy]$ ]]
    then
        exit 1
    fi
fi

# Adjusting CUDA minor to match existing PyTorch and PyTorch Geometric
# wheels
if [[ ${CUDA_MAJOR} == 10 && ${CUDA_MINOR} == 0 ]]
then
    CUDA_MINOR_TORCH=1
    CUDA_MINOR_TV=1
    CUDA_MINOR_PYG=1
fi
if [[ ${CUDA_MAJOR} == 10 && ${CUDA_MINOR} > 1 ]]
then
    CUDA_MINOR_TORCH=2
    CUDA_MINOR_TV=1
    CUDA_MINOR_PYG=2
fi
if [[ ${CUDA_MAJOR} == 11 ]]
then
    CUDA_MINOR_TORCH=0
    CUDA_MINOR_TV=0
    CUDA_MINOR_PYG=0
fi

# Summing up used CUDA versions
cuXXX_TORCH=`echo "cu"${CUDA_MAJOR}${CUDA_MINOR_TORCH}`
cuXXX_TV=`echo "cu"${CUDA_MAJOR}${CUDA_MINOR_TV}`
cuXXX_PYG=`echo "cu"${CUDA_MAJOR}${CUDA_MINOR_PYG}`
echo "CUDA version installed: ${CUDA_MAJOR}.${CUDA_MINOR}"
echo "CUDA version used for Torch: ${CUDA_MAJOR}.${CUDA_MINOR_TORCH}"
echo "CUDA version used for TorchVision: ${CUDA_MAJOR}.${CUDA_MINOR_TV}"
echo "CUDA version used for Torch Geometric: ${CUDA_MAJOR}.${CUDA_MINOR_PYG}"
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
# See https://pytorch.org/get-started/previous-versions/ if wheel issues
pip install torch==${TORCH}+${cuXXX_TORCH} torchvision==0.8.2+${cuXXX_TV} --no-cache-dir -f https://download.pytorch.org/whl/torch_stable.html
pip install torch==${TORCH}+${cu110} --no-cache-dir -f https://download.pytorch.org/whl/torch_stable.html
pip install torchvision==0.8.2+${cuXXX_TV} --no-cache-dir -f https://download.pytorch.org/whl/torch_stable.html
pip install torchnet

# Install torch-geometric and dependencies
# See https://pytorch-geometric.com/whl/ if wheel issues
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${cuXXX_PYG}.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${cuXXX_PYG}.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${cuXXX_PYG}.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${cuXXX_PYG}.html
pip install torch-geometric==1.7.2
pip install torch-points-kernels==0.6.10 --no-cache-dir

# Additional dependencies
pip install omegaconf
pip install wandb
pip install tensorboard
pip install plyfile
pip install hydra-core==1.1.0
pip install pytorch-metric-learning
pip install matplotlib
pip install seaborn
pip install pykeops==1.4.2
pip install imageio
pip install opencv-python
pip install pypng
pip install git+http://github.com/CSAILVision/semantic-segmentation-pytorch.git@master
pip install h5py
pip install faiss-gpu===1.6.5

# Install MinkowskiEngine
sudo apt install libopenblas-dev
pip install -U MinkowskiEngine==v0.4.3 --install-option="--blas=openblas" -v --no-deps

# Install torchsparse
sudo apt-get install libsparsehash-dev
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0 #####################

# Install plotly and associated jupyter requirements
pip install plotly==5.4.0
pip install "jupyterlab>=3" "ipywidgets>=7.6"
pip install jupyter-dash
jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget@4.14.1
echo "NB: In case you want to use our notebooks, make sure the 'deep_view_aggregation' kernel is accessible in jupyterlab"
echo
echo

echo "Successfully installed Deep View Aggregation."

echo
echo
