### Install CUDA 10.2

### Install conda

# Git install torch-points3d
mkdir projects
git clone https://gitlab.csai.myengie.com/ia/core/3d/torch-points3d.git
cd torch-points3d

# Create tp3d-dev from yml
conda env create -f tp3d-dev.yml  ### includes the torch installation

# Activate the env
source ~/anaconda3/etc/profile.d/conda.sh  ### make sure this is the path to conda
conda activate tp3d-dev

# Install MinkowskiEngine
sudo apt install libopenblas-dev
pip install -U MinkowskiEngine --install-option="--blas=openblas" -v --no-deps

# Install torchsparse
sudo apt-get install libsparsehash-dev
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git

# Install plotly for jupyter
conda install jupyterlab "ipywidgets=7.5"
jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget@4.14.1

### Make tp3d-dev kernels accessible on jupyterlab
