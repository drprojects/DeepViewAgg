
import os
import sys
from omegaconf import OmegaConf


if __name__ == "__main__":
    # DIR = os.path.dirname(os.getcwd())
    DIR = "torch_points3d"
    ROOT = os.path.join(DIR, "..")
    sys.path.insert(0, ROOT)
    sys.path.insert(0, DIR)
    from torch_points3d.datasets.segmentation.multimodal import S3DISFusedDataset

    from time import time
    start = time()

    dataset_options = OmegaConf.load(os.path.join('conf/data/segmentation/multimodal/s3disfused.yaml'))
    # Set root to the DATA drive, where the data was downloaded
    # dataset_options.data.dataroot = "/mnt/fa444ffd-fdb4-4701-88e7-f00297a8e29b/projects/datasets/s3dis_multimodal"
    dataset_options.data.dataroot = "/media/drobert-admin/DATA/datasets/s3dis_tp3d_multimodal"
    dataset = S3DISFusedDataset(dataset_options.data)
    
    print("Dataset")
    print(dataset)
    print("-" * 50 + "\n")

    print()
    print(f"Total preprocessing time: {time() - start:0.0f}")
