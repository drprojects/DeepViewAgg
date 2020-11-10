
import os
import sys
import panel as pn
import numpy as np
import pyvista as pv
import glob
from matplotlib.colors import ListedColormap
from omegaconf import OmegaConf


if __name__ == "__main__":
    # DIR = os.path.dirname(os.getcwd())
    DIR = "torch_points3d"
    ROOT = os.path.join(DIR, "..")
    sys.path.insert(0, ROOT)
    sys.path.insert(0, DIR)
    from torch_points3d.datasets.multimodal.segmentation.s3dis import S3DISFusedDataset


    dataset_options = OmegaConf.load(os.path.join('conf/data/segmentation_multimodal/s3disfused.yaml'))
    # Set root to the DATA drive, where the data was downloaded
    # dataset_options.data.dataroot = "/mnt/fa444ffd-fdb4-4701-88e7-f00297a8e29b/projects/datasets/s3dis_multimodal"
    dataset_options.data.dataroot = "/media/drobert-admin/DATA/datasets/s3dis_subset_tp3d_multimodal"
    dataset = S3DISFusedDataset(dataset_options.data)



    print("Dataset")
    print(dataset)
    print("-" * 50 + "\n")

