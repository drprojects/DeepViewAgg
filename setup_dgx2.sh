### Requirements:
# conda
# cuda >= 10.2
# gcc >= 7
# torch >= 1.8
# Plotly >= 4.14

# Local variables
CONDA_DIR=~/anaconda3  ### Correct path to source conda on DGX-2 ?
PROJECT_DIR=/home/WL5719/workspace/projects/torch-points3d
YML_FILE=${PROJECT_DIR}/tp3d_dev.yml

# Git install torch-points3d
cd ${PROJECTS_DIR}

# Create tp3d-dev from yml
conda env create -f ${YML_FILE}

# Activate the env
source ${CONDA_DIR}/etc/profile.d/conda.sh  
conda activate tp3d_dev

# Install MinkowskiEngine
sudo apt install libopenblas-dev
pip install -U MinkowskiEngine --install-option="--blas=openblas" -v --no-deps

# Install torchsparse
sudo apt-get install libsparsehash-dev
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git

# Install plotly for jupyter
conda install jupyterlab "ipywidgets=7.5"
jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget@4.14.1

### Make tp3d_dev kernel accessible on jupyterlab
