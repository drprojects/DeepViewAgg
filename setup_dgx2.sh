### Prerequisites:
# conda
# cuda == 10.2
# gcc >= 7
# torch == 1.7.0
# Plotly >= 4.14

# Local variables
CONDA_DIR=/opt/conda  ### Correct path to source conda on DGX-2 ?
PROJECT_DIR=/home/WL5719/workspace/projects/torch-points3d
YML_FILE=${PROJECT_DIR}/tp3d_dev.yml
CUDA=cu102
TORCH=1.7.0

# Git install torch-points3d
cd ${PROJECT_DIR}

# Create tp3d-dev from yml
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
pip install hydra-core==0.11.3
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

### Make tp3d_dev kernel accessible on jupyterlab
