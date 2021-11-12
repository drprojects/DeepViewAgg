from torch_points3d.datasets.segmentation.kitti360 import *
from torch_points3d.core.multimodal.image import SameSettingImageData
from torch_points3d.core.multimodal.data import MMData

DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)


########################################################################
#                                 Utils                                #
########################################################################

def read_kitti360_image_sequence(root, sequence, cam_id=0, size=None):
    """Read the raw KITTI360 image data for a given sequence and build a
    `SameSettingImageData` object gathering the paths and extrinsic and
    intrinsic parameters of the images.

    NB: because carelessly loading images into memory is costly, only
    meta information is computed here. Images can be loaded later on
    using `SameSettingImageData.load()`.
    """
    # Initialize paths to useful files and directories
    camera_names = ['image_00', 'image_01']
    raw_2d_dir = osp.join(root, 'data_2d_raw')
    pose_dir = osp.join(root, 'data_poses', sequence)
    calib_dir = osp.join(root, 'calibration')
    intrinsic_file = osp.join(calib_dir, 'perspective.txt')
    pose_file = osp.join(pose_dir, 'poses.txt')
    cam_to_pose_file = osp.join(calib_dir, 'calib_cam_to_pose.txt')

    # Camera-to-pose calibration
    cam_to_pose = torch.from_numpy(
        load_calibration_camera_to_pose(cam_to_pose_file)[camera_names[cam_id]])

    # System poses (different from camera pose)
    poses = np.loadtxt(pose_file)
    frames = sorted([f'{x:010d}.png' for x in poses[:, 0].astype(np.int64)])
    poses = torch.from_numpy(poses[:, 1:]).view(-1, 3, 4)
    n_images = poses.shape[0]

    # Recover the absolute path to each frame
    paths = np.asarray([
        osp.join(raw_2d_dir, sequence, camera_names[cam_id], 'data_rect', x)
        for x in frames])

    # Sanity check just to make sure all images exist
    missing = [x for x in paths if not osp.exists(x)]
    assert len(missing) > 0, \
        f'The following images could not be found:\n{missing}'

    # Intrinsic parameters
    K, R_rect, width, height = load_intrinsics(intrinsic_file, cam_id=cam_id)
    fx = torch.Tensor([K[0, 0]]).repeat(n_images)
    fy = torch.Tensor([K[1, 1]]).repeat(n_images)
    mx = torch.Tensor([K[0, 2]]).repeat(n_images)
    my = torch.Tensor([K[1, 2]]).repeat(n_images)
    R_inv = torch.from_numpy(R_rect).inverse()
    size = (width, height) if size is None else size

    # Recover the cam_to_world from system pose and calibration
    cam_to_world = []
    for pose in poses:
        pose = torch.cat((pose, torch.ones(1, 4)), dim=0)
        if cam_id == 0:
            cam_to_world.append(pose @ cam_to_pose @ R_inv)
        else:
            raise NotImplementedError(f"Unknown Camera ID '{cam_id}'!")
    cam_to_world = torch.cat([e.unsqueeze(0) for e in cam_to_world], dim=0)

    # Recover the image positions from the extrinsic matrix
    T = cam_to_world[:, :3, 3]

    # Gather the sequence image information in a `SameSettingImageData`
    images = SameSettingImageData(
        ref_size=size, proj_upscale=1, path=paths, pos=T, fx=fx, fy=fy,
        mx=mx, my=my, extrinsic=cam_to_world)

    return images


def load_intrinsics(intrinsic_file, cam_id=0):
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
    assert (intrinsic_loaded == True)
    assert (width > 0 and height > 0)

    return K, R_rect, width, height


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
    # TODO: parameters
    """

    def __init__(
            self, root, split="train", sample_per_epoch=15000, radius=6,
            sample_res=0.3, transform=None, pre_transform=None,
            pre_filter=None, keep_instance=False,
            pre_transform_image=None, transform_image=None, image_r_max=20,
            image_ratio=5, image_size=(1408, 376)):

        # Initialization with downloading and all preprocessing
        super().__init__(
            root, split=split, sample_per_epoch=sample_per_epoch,
            radius=radius, sample_res=sample_res, transform=transform,
            pre_transform=pre_transform, pre_filter=pre_filter,
            keep_instance=keep_instance)

        # 2D-related attributes
        self.pre_transform_image = pre_transform_image
        self.transform_image = transform_image
        self._image_r_max = image_r_max
        self._image_ratio = image_ratio
        self._image_size = image_size

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
        """The size of images."""
        return self._image_size

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
    def raw_file_names(self):
        """The filepaths to find in order to skip the download."""
        return super().raw_file_names + [
            'data_2d_raw', 'data_2d_test', 'data_poses', 'calibration']

    @property
    def processed_2d_file_names(self):
        suffix = '_'.join([
            f'size-{self.image_size[0]}x{self.image_size[1]}',
            f'_ratio-{self.image_ratio}',
            f'_rmax-{self.image_r_max}'])
        return [
            osp.join(self.split, '2d', f'{w}_{suffix}.pt')
            for w in self.windows]

    @property
    def processed_file_names(self):
        """The name of the files to find in the :obj:`self.processed_dir`
        folder in order to skip the processing
        """
        return super().processed_file_names + self.processed_2d_file_names

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

            # If window image data already exists, while either the
            # window data or sampling is missing, remove it, because it
            # may be out-of-date
            if osp.exists(image_path) and any([
                    not osp.exists(p) for p in [window_path, sampling_path]]):
                os.remove(image_path)

            # Extract useful information from <path>
            split, modality, sequence_name, window_name = \
                osp.splitext(window_path)[0].split('/')[:-4]

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

            # Run image pre-transform
            if self.pre_transform_image is not None:
                _, images = self.pre_transform_image(data, images)

            # Save scan 2D data to a file that will be loaded when
            # __get_item__ is called
            torch.save(images, image_path)

    def _load_window(self, idx):
        """Load a window, its sampling data, images and mappings into
        memory based on its index in `self.windows`.
       """
        # Check if the window is not already loaded
        if self.window_idx == idx:
            return

        # Load the window data and associated sampling data
        self._window = WindowMM(
            self.paths[idx], self.sampling_paths[idx], self.image_paths[idx])
        self._window_idx = idx

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

        # Recover all images and mappings from the window
        images = self.window.images

        # Run image transforms
        if self.transform_image is not None:
            data, images = self.transform_image(data, images)

        return MMData(data, image=images)


########################################################################
#                         MiniKITTI360Cylinder                         #
########################################################################

class MiniKITTI360CylinderMM(KITTI360CylinderMM):
    """A mini version of KITTI360CylinderMM with only a few windows for
    experimentation.
    """
    _WINDOWS = {k: v[:2] for k, v in WINDOWS.items()}


########################################################################
#                            KITTI360Dataset                           #
########################################################################

class KITTI360Dataset(BaseDataset):
    """
    # TODO: comments
    """
    INV_OBJECT_LABEL = INV_OBJECT_LABEL

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        cls = MiniKITTI360CylinderMM if dataset_opt.get('mini', False) \
            else KITTI360CylinderMM
        radius = dataset_opt.get('radius')
        train_sample_res = dataset_opt.get('train_sample_res', 0.3)
        eval_sample_res = dataset_opt.get('eval_sample_res', radius / 2)
        keep_instance = dataset_opt.get('keep_instance', False)
        image_r_max = dataset_opt.get('image_r_max')
        image_ratio = dataset_opt.get('image_ratio')
        image_size = tuple(dataset_opt.get('resolution_2d'))
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
            sample_per_epoch=sample_per_epoch,
            split='train' if not train_is_trainval else 'trainval',
            pre_transform=self.pre_transform,
            transform=self.train_transform)

        self.val_dataset = cls(
            self._data_path,
            radius=radius,
            sample_res=eval_sample_res,
            keep_instance=keep_instance,
            image_r_max=image_r_max,
            image_ratio=image_ratio,
            image_size=image_size,
            sample_per_epoch=-1,
            split='val',
            pre_transform=self.pre_transform,
            transform=self.val_transform)

        self.test_dataset = cls(
            self._data_path,
            radius=radius,
            sample_res=eval_sample_res,
            keep_instance=keep_instance,
            image_r_max=image_r_max,
            image_ratio=image_ratio,
            image_size=image_size,
            sample_per_epoch=-1,
            split='test',
            pre_transform=self.pre_transform,
            transform=self.test_transform)

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
