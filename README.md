# DeepViewAgg [CVPR 2022 Oral]
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/learning-multi-view-aggregation-in-the-wild/semantic-segmentation-on-s3dis)](https://paperswithcode.com/sota/semantic-segmentation-on-s3dis?p=learning-multi-view-aggregation-in-the-wild) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/learning-multi-view-aggregation-in-the-wild/3d-semantic-segmentation-on-kitti-360)](https://paperswithcode.com/sota/3d-semantic-segmentation-on-kitti-360?p=learning-multi-view-aggregation-in-the-wild)

Official repository for **_Learning Multi-View Aggregation In the Wild for Large-Scale 3D Semantic Segmentation_** [paper :page_facing_up:](http://arxiv.org/abs/2204.07548) selected for an Oral presentation at CVPR 2022.

<p align="center">
  <img width="40%" height="40%" src="./illustrations/teaser.png">
</p>

*We propose to exploit the synergy between images and 3D point clouds by learning to select the most relevant views for each point. Our approach uses the viewing conditions of 3D points to merge features from images taken at arbitrary positions. We reach SOTA results for S3DIS (74.7 mIoU 6-Fold) and on KITTI- 360 (58.3 mIoU) without requiring point colorization, meshing, or the use of depth cameras: our full pipeline only requires raw 3D scans and a set of images and poses.*

## Coming very soon :rotating_light: :construction:
- **notebooks** for manipulating multimodal data for S3DIS, ScanNet and KITTI-360, training and testing models and reproducing our papers' main results.
- **pretrained weights** from our best-performing model on S3DIS and KITTI-360
- **[wandb](https://wandb.ai) logs** of our experiments

## Requirements :memo:
The following must be installed before installing this project.
- Anaconda3
- cuda >= 10.1
- gcc >= 7

All remaining dependencies (PyTorch, PyTorch Geometric, etc) should be installed using the provided [installation script](install.sh).

The code has been tested in the following environment:
- Ubuntu 18.04.6 LTS
- Python 3.8.5
- PyTorch 1.7.1
- CUDA 10.2, 11.2 and 11.4

## Installation :bricks:
To install DeepViewAgg, simply run `./install.sh` from inside the repository. 
- You will need to have **sudo rights** to install [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) and [TorchSparse](https://github.com/mit-han-lab/torchsparse) dependencies.
- :warning: **Do not** install Torch-Points3D from the official repository, or with `pip`.

## Disclaimer
This is **not the official [Torch-Points3D](https://github.com/nicolas-chaulet/torch-points3d) framework**. This work builds on and modifies a fixed version of the framework and has not been merged with the official repository yet. In particular, this repository **introduces numerous features for multimodal learning on large-scale 3D point clouds**. In this repository, some TP3D-specific files were removed for simplicity. 

## Project structure
The project follows the original [Torch-Points3D framework](https://github.com/nicolas-chaulet/torch-points3d) structure.
```bash
├─ conf                    # All configurations live there
├─ notebooks               # Notebooks to get started with multimodal datasets and models
├─ eval.py                 # Eval script
├─ insall.sh               # Installation script for DeepViewAgg
├─ scripts                 # Some scripts to help manage the project
├─ torch_points3d
    ├─ core                # Core components
    ├─ datasets            # All code related to datasets
    ├─ metrics             # All metrics and trackers
    ├─ models              # All models
    ├─ modules             # Basic modules that can be used in a modular way
    ├─ utils               # Various utils
    └─ visualization       # Visualization
└─ train.py                # Main script to launch a training
```

Several changes were made to extend the original project to multimodal learning on point clouds with images. 
The most important ones can be found in the following:
- `conf/data/segmentation/multimodal`: configs for the 3D+2D datasets.
- `conf/models/segmentation/multimodal`: configs for the 3D+2D models.
- `torch_points3d/core/data_transform/multimodal`: transforms for 3D+2D data.
- `torch_points3d/core/multimodal`: multimodal data and mapping objects.
- `torch_points3d/datasets/segmentation/multimodal`: 3D+2D datasets (eg S3DIS, ScanNet, KITTI360).
- `torch_points3d/models/segmentation/multimodal`: 3D+2D architectures.
- `torch_points3d/modules/multimodal`: 3D+2D modules. This is where the DeepViewAgg module can be found.
- `torch_points3d/visualization/multimodal_data.py`: tools for interactive visualization of multimodal data.

## Getting started :rocket:
Notebooks available very soon :rotating_light: :construction:

## Documentation :books:
The official documentation of [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html) and [Torch-Points3D](https://torch-points3d.readthedocs.io/en/latest/index.html#) are good starting points, since this project largely builds on top of these frameworks. For DeepViewAgg-specific features (*i.e.* all that concerns multimodal learning), the provided code is commented as much as possible, but hit me up :speech_balloon: if some parts need clarification.

## Visualization of multimodal data :telescope:
We provide code to produce interactive and sharable HTML visualizations of multimodal data and point-image mappings:

<p align="center">
  <img width="60%" height="60%" src="./illustrations/interactive_visualization_snapshot.png">
</p>
 
Examples of such HTML produced on S3DIS Fold 5 are zipped [here](./illustrations/interactive_visualizations.zip) and can be opened in your browser.

## Credits :credit_card:
- This implementation of DeepViewAgg largely relies on the [Torch-Points3D framework](https://github.com/nicolas-chaulet/torch-points3d), although not merged with the official project at this point. 
- For datasets, some code from the official [KITTI-360](https://github.com/autonomousvision/kitti360Scripts) and [ScanNet](https://github.com/ScanNet/ScanNet) repositories was used.

## Reference
In case you use all or part of the present code, please include a citation to the following paper:

```
@inproceedings{robert2022dva,
  title={Learning Multi-View Aggregation In the Wild for Large-Scale 3D Semantic Segmentation},
  author={Robert, Damien and Vallet, Bruno and Landrieu, Loic},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022},
  url = {\url{https://github.com/drprojects/DeepViewAgg}}
}
```
