
import os
import sys
import numpy as np
import torch
import glob
from matplotlib.colors import ListedColormap
from omegaconf import OmegaConf
from time import time

_ = torch.cuda.is_available()
_ = torch.cuda.memory_allocated()


if __name__ == "__main__":
    # DIR = os.path.dirname(os.getcwd())
    DIR = "torch_points3d"
    ROOT = os.path.join(DIR, "..")
    sys.path.insert(0, ROOT)
    sys.path.insert(0, DIR)
    from torch_points3d.datasets.segmentation.multimodal import S3DISFusedDataset

    start = time()
    dataset = torch.load('/home/ign.fr/drobert-admin/Bureau/s3dis_test_bckp.pt')
    print(f"Time = {time() - start:0.1f} sec.")

    n_iter = 200
    start = time()
    for _ in range(n_iter):
        _ = dataset.train_dataset[0]
    time_per_iter = (time() - start) / n_iter
    print(f"Time per iteration: {time_per_iter:0.4f}")
