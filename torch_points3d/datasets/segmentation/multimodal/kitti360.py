import re
import yaml
from sys import exit
from torch_points3d.datasets.segmentation.kitti360 import *
from torch_points3d.datasets.base_dataset_multimodal import BaseDatasetMM
from torch_points3d.core.multimodal.image import SameSettingImageData
from torch_points3d.core.multimodal.data import MMData
from torch_points3d.core.data_transform.multimodal.image import \
    DropImagesOutsideDataBoundingBox, PickKImages

import torch_points3d
TP3D_DIR = osp.dirname(osp.dirname(osp.abspath(torch_points3d.__file__)))

DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)
log = logging.getLogger(__name__)


########################################################################
#                                 Utils                                #
########################################################################

def readYAMLFile(fileName):
    '''make OpenCV YAML file compatible with python'''
    ret = {}
    skip_lines = 1  # Skip the first line which says "%YAML:1.0". Or replace it with "%YAML 1.0"
    with open(fileName) as fin:
        for i in range(skip_lines):
            fin.readline()
        yamlFileOut = fin.read()
        myRe = re.compile(r":([^ ])")  # Add space after ":", if it doesn't exist. Python yaml requirement
        yamlFileOut = myRe.sub(r': \1', yamlFileOut)
        ret = yaml.load(yamlFileOut)
    return ret


def read_kitti360_image_sequence(root, sequence, cam_id=0, size=None):
    """Read the raw KITTI360 image data for a given sequence and build a
    `SameSettingImageData` object gathering the paths and extrinsic and
    intrinsic parameters of the images.

    NB: because carelessly loading images into memory is costly, only
    meta information is computed here. Images can be loaded later on
    using `SameSettingImageData.load()`.
    """
    # Check on the camera id
    if cam_id not in range(4):
        NotImplementedError(f"Unknown Camera ID '{cam_id}'!")
    fisheye = cam_id >= 2

    # Initialize paths to useful files and directories
    camera_names = ['image_00', 'image_01', 'image_02', 'image_03']
    raw_2d_dir = osp.join(root, 'data_2d_raw')
    pose_dir = osp.join(root, 'data_poses', sequence)
    calib_dir = osp.join(root, 'calibration')
    pose_file = osp.join(pose_dir, 'poses.txt')
    cam_to_pose_file = osp.join(calib_dir, 'calib_cam_to_pose.txt')

    # Camera-to-pose calibration
    cam_to_pose = torch.from_numpy(
        load_calibration_camera_to_pose(cam_to_pose_file)[camera_names[cam_id]])

    # System poses (different from camera pose)
    poses = np.loadtxt(pose_file)
    frames = sorted([f'{x:010d}.png' for x in poses[:, 0].astype(np.int64)])
    poses = torch.from_numpy(poses[:, 1:]).view(-1, 3, 4)

    # Recover the absolute path to each frame
    paths = np.asarray([
        osp.join(raw_2d_dir, sequence, camera_names[cam_id], 'data_rect', x)
        for x in frames])

    # The poses may describe images that are not provided, so remove
    # missing images for paths and poses
    idx = np.array([i for i, p in enumerate(paths) if osp.exists(p)]).astype(np.int64)
    poses = poses[torch.from_numpy(idx)]
    paths = paths[idx]
    n_images = poses.shape[0]
    if n_images == 0:
        raise ValueError(f'Could not fin any image')

    # Intrinsic parameters
    if not fisheye:
        intrinsic_file = osp.join(calib_dir, 'perspective.txt')
        K, R_rect, width, height = load_intrinsics_perspective(
            intrinsic_file, cam_id=cam_id)
        fx = torch.Tensor([K[0, 0]]).repeat(n_images)
        fy = torch.Tensor([K[1, 1]]).repeat(n_images)
        mx = torch.Tensor([K[0, 2]]).repeat(n_images)
        my = torch.Tensor([K[1, 2]]).repeat(n_images)
        R_inv = torch.from_numpy(R_rect).inverse()
    else:
        intrinsic_file = osp.join(calib_dir, f'image_0{cam_id}.yaml')
        fi, width, height = load_intrinsics_fisheye(intrinsic_file)
        xi = torch.Tensor(fi['mirror_parameters']['xi']).repeat(n_images)
        k1 = torch.Tensor(fi['distortion_parameters']['k1']).repeat(n_images)
        k2 = torch.Tensor(fi['distortion_parameters']['k2']).repeat(n_images)
        gamma1 = torch.Tensor(fi['projection_parameters']['gamma1']).repeat(n_images)
        gamma2 = torch.Tensor(fi['projection_parameters']['gamma2']).repeat(n_images)
        u0 = torch.Tensor(fi['projection_parameters']['u0']).repeat(n_images)
        v0 = torch.Tensor(fi['projection_parameters']['v0']).repeat(n_images)
    size = (width, height) if size is None else size

    # Recover the cam_to_world from system pose and calibration
    cam_to_world = []
    for pose in poses:
        pose = torch.cat((pose, torch.ones(1, 4)), dim=0)
        if not fisheye:
            cam_to_world.append(pose @ cam_to_pose @ R_inv)
        else:
            cam_to_world.append(pose @ cam_to_pose)
    cam_to_world = torch.cat([e.unsqueeze(0) for e in cam_to_world], dim=0)

    # Recover the image positions from the extrinsic matrix
    T = cam_to_world[:, :3, 3]

    # Gather the sequence image information in a `SameSettingImageData`
    if not fisheye:
        images = SameSettingImageData(
            ref_size=size, proj_upscale=1, path=paths, pos=T, fx=fx, fy=fy,
            mx=mx, my=my, extrinsic=cam_to_world)
    else:
        images = SameSettingImageData(
            ref_size=size, proj_upscale=1, path=paths, pos=T, xi=xi, k1=k1,
            k2=k2, gamma1=gamma1, gamma2=gamma2, u0=u0, v0=v0,
            extrinsic=cam_to_world)

    return images


def load_intrinsics_perspective(intrinsic_file, cam_id=0):
    """Load KITTI360 perspective camera intrinsics.

    Credit: https://github.com/autonomousvision/kitti360Scripts
    """
    intrinsic_loaded = False
    width = -1
    height = -1
    with open(intrinsic_file) as f:
        intrinsics = f.read().splitlines()
    for line in intrinsics:
        line = line.split(' ')
        if line[0] == f'P_rect_0{cam_id}:':
            K = [float(x) for x in line[1:]]
            K = np.reshape(K, [3, 4])
            intrinsic_loaded = True
        elif line[0] == f'R_rect_0{cam_id}:':
            R_rect = np.eye(4)
            R_rect[:3, :3] = np.array([
                float(x) for x in line[1:]]).reshape(3, 3)
        elif line[0] == f"S_rect_0{cam_id}:":
            width = int(float(line[1]))
            height = int(float(line[2]))
    assert intrinsic_loaded
    assert width > 0 and height > 0

    return K, R_rect, width, height


def load_intrinsics_fisheye(intrinsic_file):
    """Load KITTI360 fisheye camera intrinsics.

    Credit: https://github.com/autonomousvision/kitti360Scripts
    """
    intrinsics = readYAMLFile(intrinsic_file)
    width, height = intrinsics['image_width'], intrinsics['image_height']
    fi = intrinsics
    return fi, width, height


def load_calibration_camera_to_pose(filename):
    """Load KITTI360 camera-to-pose calibration.

    Credit: https://github.com/autonomousvision/kitti360Scripts
    """
    Tr = {}
    with open(filename, 'r') as fid:
        cameras = ['image_00', 'image_01', 'image_02', 'image_03']
        lastrow = np.array([0, 0, 0, 1]).reshape(1, 4)
        for camera in cameras:
            Tr[camera] = np.concatenate(
                (read_variable(fid, camera, 3, 4), lastrow))
    return Tr


########################################################################
#                                Window                                #
########################################################################

class WindowMM(Window):
    """Small placeholder for point cloud and images window data."""

    def __init__(self, window_path, sampling_path, image_path):
        # Load point clouds and sampling
        super().__init__(window_path, sampling_path)

        # Load window images and mappings
        self._images = torch.load(image_path)

    @property
    def images(self):
        return self._images

    @property
    def num_views(self):
        return self.images.num_views

    def __repr__(self):
        display_attr = [
            'split', 'sequence', 'window', 'num_points', 'num_centers',
            'num_views']
        attr = ', '.join([f'{a}={getattr(self, a)}' for a in display_attr])
        return f'{self.__class__.__name__}({attr})'


########################################################################
#                             Window Buffer                            #
########################################################################

class WindowMMBuffer(WindowBuffer):
    """Takes care of loading and discarding windows for us. Since we
    can't afford loading all windows at once in memory, the
    `WindowBuffer` keeps at most `size` windows loaded at a time. When
    an additional window is queried, the buffer is updated in a
    first-in-first-out fashion.
    """

    def __getitem__(self, idx):
        """Load a window into memory based on its index in
        `self._dataset.windows`.
        """
        # Check if the window is not already loaded
        if idx in self.idx_loaded:
            return self._windows[idx]

        # If the buffer is full, drop the oldest window
        if self.is_full:
            self._drop_oldest_window()

        # Load the window data and associated sampling data
        self._windows[idx] = WindowMM(
            self._dataset.paths[idx], self._dataset.sampling_paths[idx],
            self._dataset.image_paths[idx])
        self._queue.append(idx)

        return self._windows[idx]


########################################################################
#                           KITTI360Cylinder                           #
########################################################################

class KITTI360CylinderMM(KITTI360Cylinder):
    """
    Child class of KITTI360Cylinder supporting sampling of 3D cylinders
    along with images and mappings for each window.

    When `sample_per_epoch` is specified, indexing the dataset produces
    cylinders randomly picked so as to even-out class distributions.
    When `sample_per_epoch=0`, the cylinders are regularly sampled and
    accessed normally by indexing the dataset.

    http://www.cvlibs.net/datasets/kitti-360/

    Parameters
    ----------
    Parameters
    ----------
    root : `str`
        Path to the root data directory.
    split : {'train', 'val', 'test', 'trainval'}, optional
    sample_per_epoch : `int`, optional
        Rules the sampling mechanism for the dataset.

        When `self.sample_per_epoch > 0`, indexing the dataset produces
        random cylindrical sampling, picked so as to even-out the class
        distribution across the dataset.

        When `self.sample_per_epoch <= 0`, indexing the dataset
        addresses cylindrical samples in a deterministic fashion. The
        cylinder indices are ordered with respect to their acquisition
        window and the regular grid sampling of the centers in each
        window.
    radius : `float`, optional
        The radius of cylindrical samples.
    sample_res : `float`, optional
        The resolution of the grid on which cylindrical samples are
        generated. The higher the `sample_res`, the less cylinders
        in the dataset.
    transform : callable, optional
        transform function operating on data.
    pre_transform : callable, optional
        pre_transform function operating on data.
    pre_filter : callable, optional
        pre_filter function operating on data.
    keep_instance : `bool`, optional
        Whether instance labels should be loaded.
    pre_transform_image : callable, optional
        pre_transform_image function operating on data and images
    transform_image : callable, optional
        transform_image function operating on data and images
    buffer : `int`, optional
        Number of windows the buffer can hold in memory at once.
    image_r_max : `float`, optional
        The maximum radius of image mappings.
    image_ratio : `float`, optional
        The ratio of images used. A ratio of 5 means out-every five
        images are selected from the sequence images.
    image_size : `tuple`, optional
        The size of images used in mappings.
    voxel : `float`, optional
        The voxel resolution of the point clouds used in mappings.
    """

    def __init__(
            self, root, split="train", sample_per_epoch=15000, radius=6,
            sample_res=6, transform=None, pre_transform=None,
            pre_filter=None, keep_instance=False, pre_transform_image=None,
            transform_image=None, buffer=3, image_r_max=20, image_ratio=5,
            image_size=(1408, 376), voxel=0.05, cam_id=0):

        # 2D-related attributes
        self.pre_transform_image = pre_transform_image
        self.transform_image = transform_image
        self._image_r_max = image_r_max
        self._image_ratio = image_ratio
        self._image_size = image_size
        self._voxel = voxel
        self._cam_id = cam_id

        # Initialization with downloading and all preprocessing
        super().__init__(
            root, split=split, sample_per_epoch=sample_per_epoch,
            radius=radius, sample_res=sample_res, transform=transform,
            pre_transform=pre_transform, pre_filter=pre_filter,
            keep_instance=keep_instance, buffer=buffer)

        # Initialize the window buffer that will take care of loading
        # and dropping windows in memory
        self._buffer = WindowMMBuffer(self, buffer)

    @property
    def image_r_max(self):
        """The maximum radius of image mappings."""
        return self._image_r_max

    @property
    def image_ratio(self):
        """The ratio of images used. A ratio of 5 means out-every five
        images are selected from the sequence images.
        """
        return self._image_ratio

    @property
    def image_size(self):
        """The size of images used in mappings."""
        return self._image_size

    @property
    def voxel(self):
        """The voxel resolution of the point clouds used in mappings."""
        return self._voxel

    @property
    def cam_id(self):
        """The ID of the camera used."""
        return self._cam_id

    @property
    def sequences(self):
        """Sequences present in `self.windows`."""
        return sorted(list(set([w.split('/')[0] for w in self.windows])))

    @property
    def image_paths(self):
        """Paths to the dataset windows images and mappings."""
        return [
            osp.join(self.processed_dir, p)
            for p in self.processed_2d_file_names]

    @property
    def raw_file_structure(self):
        return f"""
    {self.root}/
        └── raw/
            ├── data_3d_semantics/
            |   └── 2013_05_28_drive_{{seq:0>4}}_sync/
            |       └── static/
            |           └── {{start_frame:0>10}}_{{end_frame:0>10}}.ply
            ├── data_2d_raw/
            |   └── 2013_05_28_drive_{{seq:0>4}}_sync/
            |       ├── image_{{00|01}}/
            |       |   └── data_rect/
            |       |       └── {{frame:0>10}}.png
            |       └── image_{{02|03}}/
            |           └── data_rgb/
            |               └── {{frame:0>10}}.png
            ├── data_poses/
            |   └── 2013_05_28_drive_{{seq:0>4}}_sync/
            |       ├── poses.txt
            |       └── cam0_to_world.txt   
            └── calibration/
                ├── calib_cam_to_pose.txt
                ├── calib_cam_to_velo.txt
                ├── calib_sick_to_velo.txt
                ├── perspective.txt
                └── image_{{02|03}}.yaml
            """

    @property
    def raw_file_names(self):
        """The filepaths to find in order to skip the download."""
        return super().raw_file_names + self.raw_file_names_2d + \
               self.raw_file_names_poses + [self.raw_file_names_calibration]

    @property
    def raw_file_names_2d(self):
        """Some of the file paths to find in order to skip the download.
        """
        return [
            osp.join('data_2d_raw', x, f'image_0{self.cam_id}')
            for x in self.sequences]

    @property
    def raw_file_names_calibration(self):
        """Some of the file paths to find in order to skip the download.
        """
        return 'calibration'

    @property
    def raw_file_names_poses(self):
        """Some of the file paths to find in order to skip the download.
        """
        return [osp.join('data_poses', x) for x in self.sequences]

    @property
    def processed_2d_file_names(self):
        # TODO: in theory, to make sure we track whenever the parameters
        #  affect the mapping, we should also track 'exact_splatting_2d',
        #  'k_swell', 'd_swell' or, even better, the whole visibility
        #  model. Not sure how hashing all this together would produce
        #  consistent hashes.
        suffix = '_'.join([
            f'voxel-{int(self.voxel * 100)}',
            f'size-{self.image_size[0]}x{self.image_size[1]}',
            f'ratio-{self.image_ratio}',
            f'rmax-{self.image_r_max}'])

        # For 'trainval', we use files from 'train' and 'val' to save
        # memory
        if self.split == 'trainval':
            return [
                osp.join(s, '2d', f'{w}_{suffix}.pt')
                for s in ('train', 'val')
                for w in self._WINDOWS[s]]

        return [
            osp.join(self.split, '2d', f'{w}_{suffix}.pt')
            for w in self.windows]

    @property
    def processed_file_names(self):
        """The name of the files to find in the :obj:`self.processed_dir`
        folder in order to skip the processing
        """
        return super().processed_file_names + self.processed_2d_file_names

    def download(self):
        self.download_warning()
        exit()

    def process(self):
        # Gather the images from each sequence
        sequence_images = {
            s: read_kitti360_image_sequence(self.raw_dir, s)
            for s in self.sequences}

        # Process each window one by one
        for wsi in tq(zip(self.paths, self.sampling_paths, self.image_paths)):

            # If required files exist, skip processing
            if all([osp.exists(p) for p in wsi]):
                continue

            # Recover the path for each type of data
            window_path, sampling_path, image_path = wsi

            # If window image data already exists, while the window data
            # is missing, remove it, because it may be out-of-date
            if osp.exists(image_path) and not osp.exists(window_path):
                os.remove(image_path)

            # Create necessary parent folders if need be
            os.makedirs(osp.dirname(image_path), exist_ok=True)

            # Extract useful information from <path>
            split, modality, sequence_name, window_name = \
                osp.splitext(window_path)[0].split('/')[-4:]

            # 3D preprocessing of window point cloud. We take the
            # opportunity that this step loads the window in memory to
            # recover it here, so as to avoid re-loading the same window
            # multiple times
            data, _ = self._process_3d(
                window_path, sampling_path, return_loaded=True)

            # Recover the sequence images. Note that there are much more
            # images than needed for the window at hand, need to select
            # images that see points in the window and discard the rest
            images = sequence_images[sequence_name]
            images.ref_size = self.image_size

            # Run hardcoded image pre-transform to:
            #   - drop images that are not close to the window at hand.
            #   Indeed, images are provided by entire sequences, so many
            #   images are far from the current window.
            #   - select 1/k images in the train and val sets and 10/k
            #   images in the test set. Indeed, the image acquisition
            #   frequency is too high for our needs in the train and val
            #   sequences. However, KITTI360 only provides about 10% of
            #   the images in the test set (witheld one are for novel
            #   view synthesis evaluation). For this reason, we try to
            #   keep 10 times more images from test than from train/val.
            t1 = DropImagesOutsideDataBoundingBox(margin=10, ignore_z=True)
            t2 = PickKImages(self.image_ratio, random=False)
            data, images = t2(*t1(data, images))

            # Run image pre-transform
            if self.pre_transform_image is not None:
                data, images = self.pre_transform_image(data, images)

            # Save scan 2D data to a file that will be loaded when
            # __get_item__ is called
            torch.save(images, image_path)

            # Save data again in case `self.pre_transform_image` has
            # modified anything
            torch.save(data, window_path)

    def __getitem__(self, idx):
        r"""Gets the cylindrical sample at index `idx` and transforms it
        (in case a `self.transform` is given).

        The expected indexing format depends on `self.is_random`. If
        `self.is_random=True` (ie `self.sample_per_epoch > 0`), then
        `idx` is a tuple carrying `(label, idx_window)` indicating
        which label to pick from which window. If `self.is_random=False`
        then `idx` is an integer in [0, len(self)-1] indicating which
        cylinder to pick among the whole dataset.

        NB: if, instead of a `(label, idx_window)` tuple, a single
        integer `idx` is passed to a `self.is_random=True` dataset,
        `__getitem__` will fallback to `self.is_random=False` behavior.
        This mechanism is required for some PyTorch Dataset core
        functionalities calling `self[0]`.
        """
        # Pick a 3D cylindrical sample and apply the 3D transforms. This
        # will take care of 'smart' window loading for us. The images
        # and mappings are loaded within `self.window`
        data = super().__getitem__(idx)

        # Recover images and mappings from the window
        images = self.buffer[int(data.idx_window)].images

        # Run image transforms
        if self.transform_image is not None:
            # TODO: do we really need to clone images here ?
            data, images = self.transform_image(data, images.clone())

        return MMData(data, image=images)


########################################################################
#                         MiniKITTI360Cylinder                         #
########################################################################

class MiniKITTI360CylinderMM(KITTI360CylinderMM):
    """A mini version of KITTI360CylinderMM with only a few windows for
    experimentation.
    """
    _WINDOWS = {k: v[:2] for k, v in WINDOWS.items()}

    # We have to include this method, otherwise the parent class skips
    # processing
    def process(self):
        super().process()

    # We have to include this method, otherwise the parent class skips
    # processing
    def download(self):
        super().download()


########################################################################
#                            KITTI360Dataset                           #
########################################################################

class KITTI360DatasetMM(BaseDatasetMM):
    """Multimodal dataset holding train, val and test sets for KITTI360.
    """

    INV_OBJECT_LABEL = INV_OBJECT_LABEL

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        cls = MiniKITTI360CylinderMM if dataset_opt.get('mini', False) \
            else KITTI360CylinderMM
        radius = dataset_opt.get('radius')
        train_sample_res = dataset_opt.get('train_sample_res', radius / 20)
        eval_sample_res = dataset_opt.get('eval_sample_res', radius)
        image_r_max = dataset_opt.get('image_r_max')
        image_ratio = dataset_opt.get('image_ratio')
        image_size = tuple(dataset_opt.get('resolution_2d'))
        voxel = dataset_opt.get('resolution_3d')
        keep_instance = dataset_opt.get('keep_instance', False)
        sample_per_epoch = dataset_opt.get('sample_per_epoch', 12000)
        train_is_trainval = dataset_opt.get('train_is_trainval', False)

        self.train_dataset = cls(
            self._data_path,
            radius=radius,
            sample_res=train_sample_res,
            keep_instance=keep_instance,
            image_r_max=image_r_max,
            image_ratio=image_ratio,
            image_size=image_size,
            voxel=voxel,
            sample_per_epoch=sample_per_epoch,
            split='train' if not train_is_trainval else 'trainval',
            pre_transform=self.pre_transform,
            transform=self.train_transform,
            pre_transform_image=self.pre_transform_image,
            transform_image=self.train_transform_image)

        self.val_dataset = cls(
            self._data_path,
            radius=radius,
            sample_res=eval_sample_res,
            keep_instance=keep_instance,
            image_r_max=image_r_max,
            image_ratio=image_ratio,
            image_size=image_size,
            voxel=voxel,
            sample_per_epoch=-1,
            split='val',
            pre_transform=self.pre_transform,
            transform=self.val_transform,
            pre_transform_image=self.pre_transform_image,
            transform_image=self.val_transform_image)
        
        # KITTI360 only provides about 10% of the images in the test set
        # images (withheld images are for novel view synthesis evaluation).
        # For this reason, the we should keep 10 times more images from
        # test than from train and val for the image distributions to be
        # comparable.
        image_ratio_test = max(int(image_ratio / 10), 1)

        self.test_dataset = cls(
            self._data_path,
            radius=radius,
            sample_res=eval_sample_res,
            keep_instance=keep_instance,
            image_r_max=image_r_max,
            image_ratio=image_ratio_test,
            image_size=image_size,
            voxel=voxel,
            sample_per_epoch=-1,
            split='test',
            pre_transform=self.pre_transform,
            transform=self.test_transform,
            pre_transform_image=self.pre_transform_image,
            transform_image=self.test_transform_image)

        # A dedicated sampler must be created for the train set. Indeed,
        # self.train_dataset.sample_per_epoch > 0 means cylindrical
        # samples will be picked randomly across all windows. In order
        # to minimize window loading overheads, the train_sampler
        # organizes the epoch batches so that same-window cylinders are
        # queried consecutively.
        self.train_sampler = KITTI360Sampler(self.train_dataset)

        # If a `class_weight_method` is provided in the dataset config,
        # the dataset will have a `weight_classes` to be used when
        # computing the loss
        if dataset_opt.class_weight_method:
            # TODO: find an elegant way of returning class weights for train set
            raise NotImplementedError('KITTI360Dataset does not support class weights yet.')

    @property
    def submission_dir(self):
        return self.train_dataset.submission_dir

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        # NB: this import needs to be here because of a loop between the
        # `dataset.segmentation.kitti360` and `metrics.kitti360_tracker`
        # modules imports
        from torch_points3d.metrics.kitti360_tracker import KITTI360Tracker
        return KITTI360Tracker(
            self, wandb_log=wandb_log, use_tensorboard=tensorboard_log,
            ignore_label=IGNORE)
