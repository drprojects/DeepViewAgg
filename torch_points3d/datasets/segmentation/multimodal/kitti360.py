from torch_points3d.datasets.segmentation.kitti360 import *

DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)


########################################################################
#                                 Utils                                #
########################################################################

def load_intrinsics(intrinsic_file, cam_id=0):
    """ Load KITTI360 perspective camera intrinsics

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
            R_rect[:3, :3] = np.array([float(x) for x in line[1:]]).reshape(3, 3)
        elif line[0] == f"S_rect_0{cam_id}:":
            width = int(float(line[1]))
            height = int(float(line[2]))
    assert (intrinsic_loaded == True)
    assert (width > 0 and height > 0)

    return K, R_rect, width, height


def load_calibration_camera_to_pose(filename):
    """ load KITTI360 camera-to-pose calibration

    Credit: https://github.com/autonomousvision/kitti360Scripts
    """
    Tr = {}
    with open(filename, 'r') as fid:
        cameras = ['image_00', 'image_01', 'image_02', 'image_03']
        lastrow = np.array([0, 0, 0, 1]).reshape(1, 4)
        for camera in cameras:
            Tr[camera] = np.concatenate((read_variable(fid, camera, 3, 4), lastrow))
    return Tr


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
    num_classes = KITTI360_NUM_CLASSES
    _WINDOWS = WINDOWS

    def __init__(
            self, root, split="train", sample_per_epoch=15000, radius=6,
            sample_res=0.3, transform=None, pre_transform=None,
            pre_filter=None, keep_instance=False,
            pre_transform_image=None, transform_image=None, image_ratio=5,
            img_size=(1408, 376)):

        # Initialization with downloading and all preprocessing
        super(KITTI360CylinderMM, self).__init__(
            root, split=split, sample_per_epoch=sample_per_epoch,
            radius=radius, sample_res=sample_res, transform=transform,
            pre_transform=pre_transform, pre_filter=pre_filter,
            keep_instance=keep_instance)

        # 2D-related attributes
        self.pre_transform_image = pre_transform_image
        self.transform_image = transform_image
        self.image_ratio = image_ratio
        self.img_size = img_size

    @property
    def raw_file_names(self):
        """The filepaths to find in order to skip the download."""
        return [
            'data_3d_semantics', 'data_3d_semantics_test', 'data_2d_raw',
            'data_2d_test', 'data_poses', 'calibration']

    @property
    def processed_2d_file_names(self):
        # TODO: ideally, would be good to indicate the mapping Rmax and
        #  visibility model in the suffix. But they are carried by the
        #  transform.
        suffix = f'{self.image_ratio}_{self.img_size[0]}x{self.img_size[1]}'
        return [
            osp.join(split, '2d', f'{p}_{suffix}.pt')
            for split, w in self._WINDOWS.items() for p in w]

    @property
    def processed_file_names(self):
        """The name of the files to find in the :obj:`self.processed_dir`
        folder in order to skip the processing
        """
        return self.processed_3d_file_names \
               + self.processed_3d_sampling_file_names \
               + self.processed_2d_file_names

    def process(self):

        # TODO
        #   For each sequence, prepare SSIDs with poses and all
        
        # TODO
        #   For each window in processed_3d_file_names, we want to take
        #   the opportunity that the window is loaded to compute the pre_transforms
        
        # TODO
        #   Careful with inplace Data modifications !!!

        # TODO: for 2D, can't simply loop over those, need to treat 2D and 3D separately
        for path in tq(self.processed_3d_file_names):

            # Extract useful information from <path>
            split, modality, sequence_name, window_name = osp.splitext(path)[0].split('/')
            window_path = osp.join(self.processed_dir, path)
            sampling_path = osp.join(
                self.processed_dir, split, modality, sequence_name,
                f'{window_name}_{hash(self._sample_res)}.pt')

            # If required files exist, skip processing
            if osp.exists(window_path) and osp.exists(sampling_path):
                continue

            # Process the window
            if not osp.exists(window_path):

                # If windows sampling data already exists, remove it,
                # because it may be out-of-date
                if osp.exists(sampling_path):
                    os.remove(sampling_path)

                # Create necessary parent folders if need be
                os.makedirs(osp.dirname(window_path), exist_ok=True)

                # Read the raw window data
                raw_3d_dir = self.raw_file_names[1] if split == 'test' \
                    else self.raw_file_names[0]
                raw_window_path = osp.join(
                    self.raw_dir, raw_3d_dir, sequence_name, 'static',
                    window_name + '.ply')
                data = read_kitti360_window(
                    raw_window_path, instance=self._keep_instance, remap=True)

                # Apply pre_transform
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                # Pre-compute KD-Tree to save time when sampling later
                tree = KDTree(np.asarray(data.pos[:, :-1]), leaf_size=10)
                data[cT.CylinderSampling.KDTREE_KEY] = tree

                # Save pre_transformed data to the processed dir/<path>
                torch.save(data, window_path)

            else:
                data = torch.load(window_path)

            # Prepare data to build cylinder centers. Only keep 'pos'
            # and 'y' (if any) attributes and drop the z coordinate in
            # 'pos'.
            # NB: we can modify 'data' inplace here to avoid cloning
            for key in data.keys:
                if key not in ['pos', 'y']:
                    delattr(data, key)
            data.pos[:, 2] = 0

            # Compute the sampling of cylinder centers for the window
            sampler = cT.GridSampling3D(size=self._sample_res)
            centers = sampler(data)
            centers.pos = centers.pos[:, :2]
            sampling = {
                'data': centers,
                'labels': None,
                'label_counts': None,
                'grid_size': self._sample_res}

            # If data has labels (ie not test set), save which labels
            # are present in the window and their count. These will be
            # used at sampling time to pick cylinders so as to even-out
            # class distributions
            if hasattr(centers, 'y'):
                unique, counts = np.unique(np.asarray(centers.y), return_counts=True)
                sampling['labels'] = unique
                sampling['label_counts'] = counts

            torch.save(sampling, sampling_path)

    def _load_window(self, idx):
        """Load a window and its sampling data into memory based on its
        index in the self.windows list.
        """
        # Check if the window is not already loaded
        if self.window_idx == idx:
            return

        # Load the window data and associated sampling data
        self._window = Window(self.paths[idx], self.sampling_paths[idx])
        self._window_idx = idx

    def __len__(self):
        return self.sample_per_epoch if self.is_random else self.sampling_sizes.sum()

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
        if self.is_random and isinstance(idx, tuple):
            data = self._get_from_label_and_window_idx(*idx)
        else:
            data = self._get_from_global_idx(idx)
        data = data if self.transform is None else self.transform(data)
        return data

    def _get_from_label_and_window_idx(self, label, idx_window):
        """Load a random cylindrical sample of label `ìdx_label` from
        window `idx_window`.
        """
        # Load the associated window
        self._load_window(idx_window)

        # Pick a random center
        valid_centers = torch.where(self.window.centers.y == label)[0]
        idx_center = np.random.choice(valid_centers.numpy())

        # Get the cylindrical sampling
        center = self.window.centers.pos[idx_center]
        sampler = cT.CylinderSampling(self._radius, center, align_origin=False)
        data = sampler(self.window.data)

        # Save the window index and center index in the data. This will
        # be used in the KITTI360Tracker to accumulate per-window votes
        data.idx_window = int(idx_window)
        data.idx_center = int(idx_center)
        data.y_center = int(label)

        return data

    def _get_from_global_idx(self, idx):
        """Load the cylindrical sample of global index `idx`. The global
        indices refer to sampling centers considered in `self.windows`
        order.
        """
        # Split the global idx into idx_window and idx_center
        cum_sizes = self.sampling_sizes.cumsum(0)
        idx_window = torch.bucketize(idx, cum_sizes, right=True)
        offsets = torch.cat((torch.zeros(1), cum_sizes)).long()
        idx_center = idx - offsets[idx_window]

        # Load the associated window
        self._load_window(idx_window)

        # Get the cylindrical sampling
        center = self.window.centers.pos[idx_center]
        y_center = self.window.centers.y[idx_center]
        sampler = cT.CylinderSampling(self._radius, center, align_origin=False)
        data = sampler(self.window.data)

        # Save the window index and center index in the data. This will
        # be used in the KITTI360Tracker to accumulate per-window votes
        data.idx_window = int(idx_window)
        data.idx_center = int(idx_center)
        data.y_center = int(y_center)

        return data

    def _pick_random_label_and_window(self):
        """Generates an `(label, idx_window)` tuple as expected by
        `self.__getitem` when `self.is_random=True`.

        This function is typically intended be used by a PyTorch Sampler
        to build a generator to iterate over random samples of the
        dataset while minimizing window loading overheads.
        """
        if not self.is_random:
            raise ValueError('Set sample_per_epoch > 0 for random sampling.')

        # First, pick a class randomly. This guarantees all classes are
        # equally represented. Note that classes are assumed to be all
        # integers in [0, self.num_classes-1] here. Besides, if a class
        # is absent from label_counts (ie no cylinder carries the
        # label), it will not be picked.
        seen_labels = torch.where(self.label_counts.sum(dim=0) > 0)[0]
        label = np.random.choice(seen_labels.numpy())

        # Then, pick a window that has a cylinder with such class, based
        # on class counts.
        counts = self.label_counts[:, label]
        weights = (counts / counts.sum()).numpy()
        idx_window = np.random.choice(range(len(self.windows)), p=weights)

        return label, idx_window


########################################################################
#                           MiniKITTI360Cylinder                           #
########################################################################

class MiniKITTI360Cylinder(KITTI360Cylinder):
    """A mini version of KITTI360Cylinder with only a few windows for
    experimentation.
    """
    _WINDOWS = {k: v[:2] for k, v in WINDOWS.items()}


########################################################################
#                              Data splits                             #
########################################################################

class KITTI360Sampler(Sampler):
    """This sampler is responsible for creating KITTICylinder
    `(label, idx_window)` indices for random sampling of cylinders
    across all windows.

    In order to minimize window loading overheads, the KITTI360Sampler
    organizes the samples so that same-window cylinders are queried
    consecutively. An optional `max_consecutive` parameter lets you
    specify the maximum tolerance on the number of consecutive samples
    from the same window. This is too avoid spending too much time
    learning from on the same 3D area.
    """

    def __init__(self, dataset, max_consecutive=40):
        # This sampler only makes sense for KITTICylinder datasets
        # implementing random sampling (ie dataset.is_random=True)
        assert dataset.is_random
        self.dataset = dataset
        self.max_consecutive = max_consecutive

    def __iter__(self):
        # Generate random (label, idx_window) tuple indices
        labels = torch.empty(len(self), dtype=torch.long)
        windows = torch.empty(len(self), dtype=torch.long)
        for i in range(len(self)):
            label, idx_window = self.dataset._pick_random_label_and_window()
            labels[i] = label
            windows[i] = idx_window

        # Shuffle the order in which required windows will be loaded
        unique_windows = windows.unique()
        window_order = unique_windows[torch.randperm(unique_windows.shape[0])]

        # Compute the order in which the cylinders will be loaded. Note
        # this disregards the max_consecutive for now
        order = window_order[windows].argsort()

        # Sort the windows and labels
        labels = labels[order]
        windows = windows[order]

        # If the max_consecutive limit is respected, end here
        if (torch.bincount(windows) <= self.max_consecutive).all():
            return iter([
                (l, w) for l, w in zip(labels.tolist(), windows.tolist())])

        # Accumulate the samplings in same-window consecutive groups of
        # the max_consecutive or less. Store the
        indices = []
        group_w = windows[0]
        group_start = 0
        for i, w in enumerate(windows):
            if w != group_w or (i - group_start) >= self.max_consecutive:
                indices.append(torch.arange(group_start, i))
                group_w = w
                group_start = i

            # Make last group 'manually'
            if i == windows.shape[0] - 1:
                indices.append(torch.arange(group_start, i+1))

        # Shuffle the groups of indices and concatenate them into the
        # final sampling order
        shuffle(indices)
        order = torch.cat(indices)

        # Sort the windows and labels to account for max_consecutive
        labels = labels[order]
        windows = windows[order]

        return iter([(l, w) for l, w in zip(labels.tolist(), windows.tolist())])

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        return f'{self.__class__.__name__}(num_samples={len(self)})'


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

        cls = MiniKITTI360Cylinder if dataset_opt.get('mini', False) else KITTI360Cylinder
        radius = dataset_opt.get('radius', 6)
        train_sample_res = dataset_opt.get('train_sample_res', 0.3)
        eval_sample_res = dataset_opt.get('eval_sample_res', radius / 2)
        keep_instance = dataset_opt.get('keep_instance', False)
        sample_per_epoch = dataset_opt.get('sample_per_epoch', 12000)
        train_is_trainval = dataset_opt.get('train_is_trainval', False)

        self.train_dataset = cls(
            self._data_path,
            radius=radius,
            sample_res=train_sample_res,
            keep_instance=keep_instance,
            sample_per_epoch=sample_per_epoch,
            split='train' if not train_is_trainval else 'trainval',
            pre_transform=self.pre_transform,
            transform=self.train_transform)

        self.val_dataset = cls(
            self._data_path,
            radius=radius,
            sample_res=eval_sample_res,
            keep_instance=keep_instance,
            sample_per_epoch=-1,
            split='val',
            pre_transform=self.pre_transform,
            transform=self.val_transform)

        self.test_dataset = cls(
            self._data_path,
            radius=radius,
            sample_res=eval_sample_res,
            keep_instance=keep_instance,
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

    @property
    def test_data(self):
        # TODO this needs to change for KITTI360, the raw data will be extracted directly from the files
        return self.test_dataset[0].raw_test_data

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        return KITTI360Tracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)
