# Do this and CLOSE THE TERMINAL
conda init bash

# In a new terminal
conda activate tp3d
PROJECT_DIR=/home/WL5719/workspace/projects/torch-points3d
cd $PROJECT_DIR

pip install hydra-core==1.1.0
pip install pykeops==1.4.2
pip install imageio
pip install opencv-python
pip install pypng
pip install git+https://github.com/CSAILVision/semantic-segmentation-pytorch.git@master

conda install -c conda-forge faiss-gpu=1.6.5