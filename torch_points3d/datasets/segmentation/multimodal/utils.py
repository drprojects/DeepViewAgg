import os.path as osp
import glob
import torch
import numpy as np
from torch_points3d.core.multimodal.image import SameSettingImageData


def read_image_pose_pairs(
        image_dir, pose_dir, image_suffix='_rgb.png',
        pose_suffix='_pose.json', skip_names=None, verbose=False):
    """
    Search for all image-pose correspondences in the directories.
    Return the list of image-pose pairs. Orphans are ignored.
    """
    # Search for images and poses
    image_names = sorted([
        osp.basename(x).replace(image_suffix, '')
        for x in glob.glob(osp.join(image_dir, '*' + image_suffix))])
    pose_names = sorted([
        osp.basename(x).replace(pose_suffix, '')
        for x in glob.glob(osp.join(pose_dir, '*' + pose_suffix))])

    # Remove images specified by skip_names
    skip_names = skip_names if skip_names is not None else []
    image_names = [x for x in image_names if x not in skip_names]
    pose_names = [x for x in pose_names if x not in skip_names]

    # Print orphans
    if not image_names == pose_names:
        image_orphan = [
            osp.join(image_dir, x + image_suffix)
            for x in set(image_names) - set(pose_names)]
        pose_orphan = [
            osp.join(pose_dir, x + pose_suffix)
            for x in set(pose_names) - set(image_names)]
        print("Could not recover all image-pose correspondences.")
        print(f"  Orphan images : {len(image_orphan)}/{len(image_names)}")
        if verbose:
            for x in image_orphan:
                print(4 * ' ' + '/'.join(x.split('/')[-4:]))
        print(f"  Orphan poses  : {len(pose_orphan)}/{len(pose_names)}")
        if verbose:
            for x in pose_orphan:
                print(4 * ' ' + '/'.join(x.split('/')[-4:]))

    # Only return the recovered pairs
    correspondences = sorted(list(set(image_names).intersection(
        set(pose_names))))
    pairs = [(
        osp.join(image_dir, x + image_suffix),
        osp.join(pose_dir, x + pose_suffix))
        for x in correspondences]
    return pairs


def img_info_to_img_data(info_ld, img_size):
    """Helper function to convert a list of image info dictionaries
    into a more convenient SameSettingImageData object.
    """
    if len(info_ld) > 0:
        info_dl = {k: [dic[k] for dic in info_ld] for k in info_ld[0]}
        image_data = SameSettingImageData(
            path=np.array(info_dl['path']), pos=torch.Tensor(info_dl['xyz']),
            opk=torch.Tensor(info_dl['opk']), ref_size=img_size)
    else:
        image_data = SameSettingImageData(ref_size=img_size)
    return image_data
