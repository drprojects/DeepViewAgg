# DeepViewAgg
Official repository for the **_Learning Multi-View Aggregation In the Wild for Large-Scale 3D Semantic Segmentation_** paper :page_facing_up:.

<p align="center">
  <img width="40%" height="40%" src="./illustrations/teaser.png">
</p>

[Paper](link-to-paper) abstract:

*Recent works on 3D semantic segmentation propose to exploit the synergy between images and point clouds by pro- cessing each modality with a dedicated network and project- ing learned 2D features onto 3D points. Merging large-scale point clouds and images raises several challenges, such as constructing a mapping between points and pixels, and ag- gregating features between multiple views. Current methods require mesh reconstruction or specialized sensors to recover occlusions, and use heuristics to select and aggregate avail- able images. In contrast, we propose an end-to-end trainable multi-view aggregation model leveraging the viewing condi- tions of 3D points to merge features from images taken at ar- bitrary positions. Our method can combine standard 2D and 3D networks and outperforms both 3D models operating on colorized point clouds and hybrid 2D/3D networks without requiring colorization, meshing, or true depth maps. We set a new state-of-the-art for large-scale indoor/outdoor semantic segmentation on S3DIS (74.7 mIoU 6-Fold) and on KITTI- 360 (58.3 mIoU). Our full pipeline only requires raw 3D scans and a set of images and poses.*

# Coming soon :rocket:

## Requirements
The following must be installed before installing this project.
- Anaconda3
- cuda >= 10.1
- gcc >= 7

All remaining dependencies (PyTorch, PyTorch Geometric, etc) should be installed using the prodived [installation script](install.sh).

The code has been tested in the following environment:
- Ubuntu 18.04.6 LTS
- Python 3.8.5
- PyTorch 1.7.1
- CUDA 10.2, 11.2 and 11.4

## Installation
To install DeepViewAgg, simply run `./install.sh` from inside the repository. 
- You will need to have **sudo rights** to install [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) and [TorchSparse](https://github.com/mit-han-lab/torchsparse) dependencies.
- **Do not** install Torch-Points3D from the official repository, nor from `pip`.

## Disclaimer
This is **not the official [Torch-Points3D framework](https://github.com/nicolas-chaulet/torch-points3d) framework**. This work builds on and modifies a fixed version of the framework and has not been merged with the official repository yet. In particular, this repository **introduces numerous features for multimodal learning on large-scale 3D point clouds**. In this repository, some TP3D-specific files were trimmed. 

## Project structure
The project follows the original [Torch-Points3D framework](https://github.com/nicolas-chaulet/torch-points3d) structure.
```bash
├─ benchmark               # Output from various benchmark runs
├─ conf                    # All configurations for training and evaluation leave there
├─ notebooks               # A collection of notebooks that allow result exploration and network debugging
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
├─ test
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

## Visualization of multimodal data
We provide code to produce interactive and sharable HTML visualizations of multimodal data and point-image mappings:

<p align="center">
  <img width="60%" height="60%" src="./illustrations/interactive_visualization_snapshot.png">
</p>
 
Examples of such HTML produced on S3DIS Fold 5 are zipped here [here](./illustrations/interactive_visualizations.zip) and can be opened in your browser.

## Credits
- This implementation of **DeepViewAgg** largely relies on the [Torch-Points3D framework](https://github.com/nicolas-chaulet/torch-points3d), although not merged with the official project at this point. 
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