### Requirements:
# conda
# cuda >= 10.2
# gcc >= 7
# torch >= 1.8
# Plotly >= 4.14

# Local variables
PROJECTS_DIR=~/workspace/projects
YML_FILE=~/workspace/tp3d_dev.yml

# Git install torch-points3d
mkdir -p ${PROJECTS_DIR}
cd ${PROJECTS_DIR}
git clone https://gitlab.csai.myengie.com/ia/core/3d/torch-points3d.git
cd torch-points3d

# Create tp3d-dev from yml
conda env create -f ${YML_FILE}

# Activate the env
source ~/anaconda3/etc/profile.d/conda.sh  ### Correct path to source conda on DGX-2 ?
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
