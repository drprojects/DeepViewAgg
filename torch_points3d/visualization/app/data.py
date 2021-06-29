import os
import sys
import numpy as np
import torch
from torch import Tensor
import glob
from matplotlib.colors import ListedColormap
from omegaconf import OmegaConf

#from torch_points3d.datasets.segmentation.multimodal.s3dis_area1_office1 import S3DISFusedDataset, OBJECT_COLOR, INV_OBJECT_LABEL
from torch_points3d.datasets.segmentation.multimodal.s3dis_area5_office40 import S3DISFusedDataset, OBJECT_COLOR, INV_OBJECT_LABEL
from torch_points3d.datasets.segmentation.multimodal import IGNORE_LABEL
from torch_points3d.visualization.multimodal_data import visualize_mm_data

from torch_geometric.data import Data, Batch

from torch_geometric.transforms import *
from torch_points3d.core.data_transform import *
from torch_points3d.core.data_transform.multimodal.image import *
from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.datasets.base_dataset_multimodal import BaseDatasetMM
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

def get_dataset():
    start = time()
    dataset_options = OmegaConf.load('conf/data/segmentation/multimodal/s3dis-area1-office1-no3d_exact_768x384.yaml')
    # dataset_options = OmegaConf.load('conf/data/segmentation/multimodal/s3dis-area5-office40-no3d_exact_768x384.yaml')

    # Set the 3D resolution
    dataset_options.data.first_subsampling = 0.05

    # Set root to the DATA drive, where the data was downloaded
    # DATA_ROOT = '/media/drobert-admin/DATA/datasets/s3dis'  # IGN DATA
    DATA_ROOT = '/media/drobert-admin/DATA2/datasets/s3dis'  # IGN DATA2
    # DATA_ROOT = '/var/data/drobert/datasets/s3dis'  # AI4GEO
    # DATA_ROOT = '/home/qt/robertda/scratch/datasets/s3dis'  # CNES
    # DATA_ROOT = '/raid/dataset/pointcloud/data/s3dis'  # ENGIE
    dataset_options.data.dataroot = os.path.join(DATA_ROOT, '5cm_exact_768x384')  # 5cm 3D + 768x384 2D

    # Build the dataset
    print('\nLoading dataset')
    dataset = S3DISFusedDataset(dataset_options.data)
    print(f"Time = {time() - start:0.2f} sec")
    print(f"Done")

    return dataset

def get_model(dataset):
    print(f"\nLoading model")

    # Load the model from checkpoint - RGB light drop 50 trained on "5cm exact 768x384"
    model_options = OmegaConf.load('conf/data/segmentation/multimodal/s3disfused-no3d_exact_768x384.yaml')
    model_options.models = OmegaConf.load('conf/models/segmentation/multimodal/no3d.yaml').models

    # model_options.model_name = 'RGB_D32-4_persistent-indrop-50_mean_view'
    # checkpoint_dir = 'outputs/benchmark/benchmark-RGB_D32-4_persistent-indrop-50_mean_view-20210517_230329'

    model_options.model_name = 'RGB_D32_gp-8-32-32-4_gscale'
    checkpoint_dir = 'outputs/benchmark/cnes/RGB/fold5/RGB_D32_gp-8-32-32-4_gscale_exact_768x384'

    selection_stage = 'val'  # train, val, test
    weight_name = 'miou'  # miou, macc, acc, ..., latest
    checkpoint = ModelCheckpoint(checkpoint_dir, model_options.model_name, selection_stage, run_config=model_options, resume=False)

    # Load multimodal model
    # model = instantiate_model(model_options, dataset)  # random model initialization
    model = checkpoint.create_model(dataset, weight_name=weight_name)  # initialize from checkpoint
    model = model.eval().cuda()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Model: {model_options.model_name}")
    print(f"    Model parameters : {n_params / 10**6:0.1f} M")
    print("Done")

    return model

def get_mm_sample(mm_idx, mm_dataset, model):
    print(f"\nLoading multimodal sample")

    # Load multimodal sample for inference
    # WARNING: be consistent with visualization sample 2D transforms
    batch = MMBatch.from_mm_data_list([
        sample_data(mm_dataset, idx=mm_idx, drop_3d=[], drop_2d=[PickImagesFromMemoryCredit])])
    print(f"    N_points={batch.data.num_nodes}, N_views={batch.modalities['image'].num_views}")

    # Inference
    print(f"    Inference...")
    model.set_input(batch, model.device)
    _ = model(batch)
    pred = model.output.detach().cpu().argmax(dim=1)
    has_img_pred = hasattr(model.input.modalities['image'][0], 'pred')
    has_img_feat = hasattr(model.input.modalities['image'][0], 'feat')
    if has_img_pred:
        pred_mod = [im.pred.detach().cpu() for im in model.input.modalities['image']]
    if has_img_feat:
        feat_mod = [im.feat.detach().cpu() for im in model.input.modalities['image'] if hasattr(im, 'feat')]
    del batch

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
        match_found = False
        i = 0
        while not match_found and i < len(mm_data.modalities['image']):
            if has_img_pred and pred_mod[i].shape[-2:] == im.img_size[::-1]:
                im.pred = pred_mod[i]
                im.pred_l2 = torch.linalg.norm(pred_mod[i], dim=1).unsqueeze(1)
                match_found = True
            if has_img_feat and feat_mod[i].shape[-2:] == im.img_size[::-1]:
                im.feat = feat_mod[i]
                im.feat_l2 = torch.linalg.norm(feat_mod[i], dim=1).unsqueeze(1)
                match_found = True
            i += 1
    
    print(mm_data)
    
    print("Done")

    return mm_data

def compute_plotly_visualizations(mm_data):
    # Build plotly visualization objects to be used in dash
    out = {}

    out['3d'], out['2d_rgb'] = visualize_mm_data(
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

    for back in ['pred', 'pred_l2', 'feat', 'feat_l2']:
        out['2d_' + back] = visualize_mm_data(
            mm_data,
            class_names=CLASSES,
            class_colors=OBJECT_COLOR,
            error_color=[255, 0, 0],
            figsize=800,
            voxel=0.05,
            show_3d=False,
            show_2d=True,
            back=back,
            front=['map', 'rgb', 'pos', 'y', 'feat_proj'],
            show_point_error=False,
            show_view_error=False,
            alpha=2.5,
            pointsize=5,
            no_output=False)[1]

    for im_rgb, im_pred in zip(out['2d_rgb']['images'], out['2d_pred']['images']):
        im_rgb.is_tp_point = im_pred.is_tp_point
        im_rgb.is_tp_view = im_pred.is_tp_view

    return out

def compute_2d_back_front_visualization(images, i_img, i_front, i_error,
        alpha=2, error_color=[255, 0, 0]):
    # Recover the image settings from global image index
    bins = np.cumsum([im.num_views for im in images])
    i_img_1 = np.digitize(i_img, bins).item()
    i_img_2 = i_img - np.r_[0, bins][i_img_1].item()
    
    print()
    print(f"i_img: {i_img}")
    print(f"num views: {[im.num_views for im in images]}")
    print(f"i_img_1: {i_img_1}")
    print(f"i_img_2: {i_img_2}")
    print(f"unique img_2: {torch.unique(images[i_img_1].mappings.feature_map_indexing[0])}")
    
    print()

    # Recover the mapping
    idx = images[i_img_1].mappings.feature_map_indexing
    select = torch.where(idx[0] == i_img_2)[0]
    idx = (..., idx[2][select], idx[3][select])

    # Recover the image background and foreground
    viz = (images[i_img_1].background[i_img_2].float() / alpha).floor().type(torch.uint8)
    color = images[i_img_1].front[i_front][select].t()

    # Apply the foreground
    viz[idx] = color

    # Recover the error mask
    # Pointwise error
    if i_error == 1:
        is_tp = images[i_img_1].is_tp_point[select].t()
    # View-wise error
    elif i_error == 2:
        is_tp = images[i_img_1].is_tp_view[select].t()
    # No error
    else:
        return viz

    # Error color as a ByteTensor
    error_color = torch.ByteTensor(error_color).view(3, 1) \
        if error_color is not None else torch.zeros(3, 1, dtype=torch.uint8)

    # Apply the error overlay
    viz[idx] = viz[idx] * is_tp + ~is_tp * error_color

    return viz
