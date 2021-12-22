### Prerequisites:
# conda
# cuda == 11.3 or 10.2
# gcc >= 7

# Local variables
#CUDA=cu102
CUDA=cu111
#CUDA=cu113
TORCH=1.9.0
#TORCH=1.10.0
ENV_NAME=tp3d_torch-${TORCH}+${CUDA}
CONDA_DIR=/opt/conda  ### Correct path to source conda on DGX-2 ?
PROJECT_DIR=/home/WL5719/workspace/projects/torch-points3d
YML_FILE=${PROJECT_DIR}/tp3d_dev.yml

# Git install torch-points3d
cd ${PROJECT_DIR}

# Create tp3d-dev from yml
conda env create -f ${YML_FILE} -n ${ENV_NAME}

# Activate the env
source ${CONDA_DIR}/etc/profile.d/conda.sh  
conda activate ${ENV_NAME}

# Install torch
#pip install torch==${TORCH}+${CUDA} torchvision==0.11.1+${CUDA} -f https://download.pytorch.org/whl/${CUDA}/torch_stable.html --no-cache-dir
pip install torch==${TORCH}+${CUDA} torchvision==0.10.0+${CUDA} -f https://download.pytorch.org/whl/${CUDA}/torch_stable.html --no-cache-dir
#pip install torch-points-kernels
pip install torch-points-kernels==0.6.10
pip install torchnet

# Install torch-geometric and dependencies
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html --no-index --no-cache-dir
pip install torch-geometric==1.7.2  # more recent 2.0+ versions of pyg are not backward-compatible with TP3D as of 12.16.2021

# torch-points3d dependencies
pip install omegaconf
pip install wandb
pip install tensorboard
pip install plyfile
pip install hydra-core==1.1.0
pip install pytorch-metric-learning
pip install pykeops==1.4.2
pip install imageio
pip install opencv-python
pip install pypng
pip install git+http://github.com/CSAILVision/semantic-segmentation-pytorch.git@master
pip install matplotlib
pip install h5py

# Install MinkowskiEngine
sudo apt install libopenblas-dev
pip install -U MinkowskiEngine==v0.4.3 --install-option="--blas=openblas" -v --no-deps

# Install torchsparse
sudo apt-get install libsparsehash-dev
pip install --upgrade git+http://github.com/mit-han-lab/torchsparse.git

# Install plotly for jupyter
pip install plotly==5.4.0
pip install "jupyterlab>=3" "ipywidgets>=7.6"
pip install jupyter-dash
#jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget@4.14.1

# Install FAISS
#pip install faiss-gpu==1.6.5
pip install faiss-gpu

### Make tp3d_dev kernel accessible on jupyterlab
