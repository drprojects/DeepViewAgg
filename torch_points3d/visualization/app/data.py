import os
import sys
import numpy as np
import torch
from torch import Tensor
import glob
from matplotlib.colors import ListedColormap
from omegaconf import OmegaConf

from torch_points3d.datasets.segmentation.multimodal.s3dis_area1_office1 import S3DISFusedDataset, OBJECT_COLOR, INV_OBJECT_LABEL
from torch_points3d.datasets.segmentation.multimodal import IGNORE_LABEL
from torch_points3d.visualization.multimodal_data import visualize_mm_data

from torch_geometric.data import Data, Batch

from torch_geometric.transforms import *
from torch_points3d.core.data_transform import *
from torch_points3d.core.data_transform.multimodal.image import *
from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.datasets.base_dataset_multimodal import BaseDatasetMM
from torch_points3d.datasets.segmentation.multimodal.s3dis_area1_office1 import S3DISFusedDataset
from torch_points3d.models.model_factory import instantiate_model
from torch_points3d.metrics.model_checkpoint import ModelCheckpoint
from torch_points3d.core.multimodal.data import MMBatch
from time import time

CLASSES = [INV_OBJECT_LABEL[i] for i in range(13)]

TRANSFORMS_3D = [
    Center, 
    RandomNoise, 
    RandomRotate, 
    RandomScaleAnisotropic,
    RandomSymmetry, 
    DropFeature, 
    AddFeatsByKeys]

TRANSFORMS_2D = [
    ToFloatImage, 
    AddPixelHeightFeature, 
    PickImagesFromMemoryCredit]

def sample_data(tg_dataset, idx=0, drop_3d=TRANSFORMS_3D, drop_2d=TRANSFORMS_2D):
    """
    Temporarily remove the 3D and 2D transforms affecting the point
    positions and images from the dataset to better visualize points
    and images relative positions.
    """
    # Drop 3D transforms
    transform = tg_dataset.transform
    tg_dataset.transform = BaseDataset.remove_transform(transform, drop_3d)

    # Drop 2D transforms
    transform_image = tg_dataset.transform_image
    tg_dataset.transform_image = BaseDatasetMM.remove_multimodal_transform(transform_image, drop_2d)

    out = tg_dataset[idx]
    tg_dataset.transform = transform
    tg_dataset.transform_image = transform_image
    return out



start = time()
dataset_options = OmegaConf.load('conf/data/segmentation/multimodal/s3dis-area1-office1-no3d_exact_768x384.yaml')

# Set the 3D resolution
dataset_options.data.first_subsampling = 0.05

# Set the number of spheres
dataset_options.data.sample_per_epoch = 8

# Set root to the DATA drive, where the data was downloaded
# DATA_ROOT = '/mnt/fa444ffd-fdb4-4701-88e7-f00297a8e29b/projects/datasets/s3dis'  # ???
# DATA_ROOT = '/media/drobert-admin/DATA/datasets/s3dis'  # IGN DATA
DATA_ROOT = '/media/drobert-admin/DATA2/datasets/s3dis'  # IGN DATA2
# DATA_ROOT = '/var/data/drobert/datasets/s3dis'  # AI4GEO
# DATA_ROOT = '/home/qt/robertda/scratch/datasets/s3dis'  # CNES
# DATA_ROOT = '/raid/dataset/pointcloud/data/s3dis'  # ENGIE
dataset_options.data.dataroot = os.path.join(DATA_ROOT, '5cm_exact_768x384')  # 5cm 3D + 768x384 2D

# Build the dataset
print('\nLoading dataset')
dataset = S3DISFusedDataset(dataset_options.data)
print(f"Time = {time() - start}")

# MM sample settings
mm_idx = 2
mm_dataset = dataset.test_dataset[0]

# Load the model from checkpoint - RGB light drop 50 trained on "5cm exact 768x384"
model_options = OmegaConf.load('conf/data/segmentation/multimodal/s3disfused-no3d_exact_768x384.yaml')
model_options.models = OmegaConf.load('conf/models/segmentation/multimodal/no3d.yaml').models
model_options.model_name = 'RGB_D32-4_persistent-indrop-50_mean_view'
checkpoint_dir = 'outputs/benchmark/benchmark-RGB_D32-4_persistent-indrop-50_mean_view-20210517_230329'
selection_stage = 'val' # train, val, test
weight_name = 'miou'  # miou, macc, acc, ..., latest
checkpoint = ModelCheckpoint(checkpoint_dir, model_options.model_name, selection_stage, run_config=model_options, resume=False)

# Load multimodal model
print(f"\nInference")
# model = instantiate_model(model_options, dataset)  # random model initialization
model = checkpoint.create_model(dataset, weight_name=weight_name)  # initialize from checkpoint
model = model.eval().cuda()
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"    Model: {model_options.model_name}")
print(f"    Model parameters : {n_params / 10**6:0.1f} M")

# Load multimodal sample for inference
# WARNING: be consistent with visualization sample 2D transforms
print(f"    Sampling multimodal data...")
batch = MMBatch.from_mm_data_list([
    sample_data(mm_dataset, idx=mm_idx, drop_3d=[], drop_2d=[PickImagesFromMemoryCredit])])
n_pixels = sum([im.num_views * im.img_size[0] * im.img_size[1] for im in batch.modalities['image']])
credit_max = dataset.train_transform_image.transforms[-1].credit
print(f"    N_points={batch.data.num_nodes}, N_views={batch.modalities['image'].num_views}")
print(f"    Used pixel credit: {n_pixels} / {credit_max} = {n_pixels / credit_max * 100:0.1f} %")
print(f"    Forward pass...")

# Inference
model.set_input(batch, model.device)
batch = model(batch)
pred = model.output.detach().cpu().argmax(dim=1)
pred_mod = [im.pred.detach().cpu() for im in model.input.modalities['image']]
del model, batch
print("    Done")

# Reload multimodal data for visualization
# WARNING: be consistent with inference sample 2D transforms
mm_data = sample_data(mm_dataset, idx=mm_idx)

# Apply 3D predictions from inference
# mm_data.data.pred = mm_data.data.y * (torch.rand(mm_data.data.y.shape) > 0.05)
mm_data.data.pred = pred

# Apply 2D predictions from inference
# Samples may not have hashed the SameSettingImageData in the same order.
# Need to dispatch image feature maps wrt img_size, assuming size is
# unique.
for im in mm_data.modalities['image']:
    for i in range(len(pred_mod)):
        if pred_mod[i].shape[-2:] == im.img_size[::-1]:
            im.pred = pred_mod[i]
            break

# Build plotly visualization objects to be used in dash
OUT_3D, OUT_RGB_2D = visualize_mm_data(
    mm_data,
    class_names=CLASSES,
    class_colors=OBJECT_COLOR,
    error_color=[255, 0, 0],
    figsize=800,
    voxel=0.05,
    show_3d=True,
    show_2d=True,
    back='x',
    front=['map', 'rgb', 'pos', 'y', 'feat_proj'],
    show_point_error=False,
    show_view_error=False,
    alpha=2.5,
    pointsize=5,
    no_output=False)

_, OUT_PRED_2D = visualize_mm_data(
    mm_data,
    class_names=CLASSES,
    class_colors=OBJECT_COLOR,
    error_color=[255, 0, 0],
    figsize=800,
    voxel=0.05,
    show_3d=False,
    show_2d=True,
    back='pred',
    front=['map', 'rgb', 'pos', 'y', 'feat_proj'],
    show_point_error=False,
    show_view_error=False,
    alpha=2.5,
    pointsize=5,
    no_output=False)

def get_2d_visualization(images, i_img, i_front, alpha=2):
    # Recover the image settings from global image index
    i_img_1 = np.digitize(i_img, np.cumsum([im.num_views for im in images])).item()
    i_img_2 = i_img - i_img_1

    # Recover the mapping
    idx = images[i_img_1].mappings.feature_map_indexing
    select = torch.where(idx[0] == i_img_2)[0]
    idx = (..., idx[2][select], idx[3][select])

    # Recover the image background and foreground
    viz = (images[i_img_1].background[i_img_2].float() / alpha).floor().type(torch.uint8)
    color = images[i_img_1].front[i_front][select].t()

    # Apply the foreground
    viz[idx] = color

    return viz
