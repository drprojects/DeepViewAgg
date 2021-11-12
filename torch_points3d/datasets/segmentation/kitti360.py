import os
import os.path as osp
import numpy as np
import torch
from plyfile import PlyData
from torch_geometric.data import InMemoryDataset, Data
from torch.utils.data import Sampler
import logging
from sklearn.neighbors import KDTree
from tqdm.auto import tqdm as tq
from random import shuffle
from datetime import datetime

import torch_points3d.core.data_transform as cT
from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.datasets.segmentation.kitti360_config import *

DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)


########################################################################
#                                 Utils                                #
########################################################################

def read_kitti360_window(
        filepath, xyz=True, rgb=True, semantic=True, instance=False,
        remap=False):
    data = Data()
    with open(filepath, "rb") as f:
        window = PlyData.read(f)
        attributes = [p.name for p in window['vertex'].properties]

        if xyz:
            data.pos = torch.stack([
                torch.FloatTensor(window["vertex"][axis])
                for axis in ["x", "y", "z"]], dim=-1)

        if rgb:
            data.rgb = torch.stack([
                torch.FloatTensor(window["vertex"][axis])
                for axis in ["red", "green", "blue"]], dim=-1) / 255

        if semantic and 'semantic' in attributes:
            y = torch.LongTensor(window["vertex"]['semantic'])
            data.y = torch.from_numpy(ID2TRAINID)[y] if remap else y

        if instance and 'instance' in attributes:
            data.instance = torch.LongTensor(window["vertex"]['instance'])

    return data


def read_variable(fid, name, M, N):
    """Credit: https://github.com/autonomousvision/kitti360Scripts"""
    # rewind
    fid.seek(0, 0)

    # search for variable identifier
    line = 1
    success = 0
    while line:
        line = fid.readline()
        if line.startswith(name):
            success = 1
            break

    # return if variable identifier not found
    if success == 0:
        return None

    # fill matrix
    line = line.replace('%s:' % name, '')
    line = line.split()
    assert (len(line) == M * N)
    line = [float(x) for x in line]
    mat = np.array(line).reshape(M, N)

    return mat


########################################################################
#                                Window                                #
########################################################################

class Window:
    """Small placeholder for point cloud window data."""

    def __init__(self, window_path, sampling_path):
        # Recover useful information from the path
        self.path = window_path
        self.sampling_path = sampling_path
        split, modality, sequence_name, window_name = osp.splitext(
            window_path)[0].split('/')[-4:]
        self.split = split
        self.modality = modality
        self.sequence = sequence_name
        self.window = window_name

        # Load window data and sampling data
        self._data = torch.load(window_path)
        self._sampling = torch.load(sampling_path)

    @property
    def data(self):
        return self._data

    @property
    def num_points(self):
        return self.data.num_nodes

    @property
    def centers(self):
        return self._sampling['data']

    @property
    def sampling_labels(self):
        return torch.from_numpy(self._sampling['labels'])

    @property
    def sampling_label_counts(self):
        return torch.from_numpy(self._sampling['label_counts'])

    @property
    def sampling_grid_size(self):
        return self._sampling['grid_size']

    @property
    def num_centers(self):
        return self.centers.num_nodes

    @property
    def num_raw_points(self):
        return self._sampling['num_raw_points']

    def __repr__(self):
        display_attr = [
            'split', 'sequence', 'window', 'num_points', 'num_centers']
        attr = ', '.join([f'{a}={getattr(self, a)}' for a in display_attr])
        return f'{self.__class__.__name__}({attr})'


########################################################################
#                           KITTI360Cylinder                           #
########################################################################

class KITTI360Cylinder(InMemoryDataset):
    """Dataset supporting sampling of 3D cylinders within each window.

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
            pre_filter=None, keep_instance=False):

        self._split = split
        self._sample_per_epoch = sample_per_epoch
        self._radius = radius
        self._sample_res = sample_res
        self._keep_instance = keep_instance
        self._window = None
        self._window_idx = None

        # Initialization with downloading and all preprocessing
        super().__init__(root, transform, pre_transform, pre_filter)

        # Read all sampling files to prepare for cylindrical sampling.
        # If self.is_random (ie sample_per_eopch > 0), need to recover
        # each window's sampling centers label counts for class-balanced
        # sampling. Otherwise, need to recover the number of cylinders
        # per window for deterministic sampling.
        self._label_counts = torch.zeros(
            len(self.windows), self.num_classes).long()
        self._sampling_sizes = torch.zeros(len(self.windows)).long()
        self._window_sizes = torch.zeros(len(self.windows)).long()
        self._window_raw_sizes = torch.zeros(len(self.windows)).long()

        for i, path in enumerate(self.sampling_paths):

            # Recover the label of cylindrical sampling centers and
            # their count in each window
            sampling = torch.load(path)
            centers = sampling['data']

            # Save the number of sampling cylinders in the window
            self._sampling_sizes[i] = centers.num_nodes

            # Save the number of points in the window, this will be
            # passed in the Data objects generated by the dataset for
            # the KITTI360Tracker to use when computing votes on
            # overlapping cylinders
            self._window_sizes[i] = sampling['num_points']
            self._window_raw_sizes[i] = sampling['num_raw_points']

            # If class-balanced sampling is not necessary, skip the rest
            if not self.is_random:
                continue

            # If the data has no labels, class-balanced sampling cannot
            # be performed
            if sampling['labels'] is None:
                raise ValueError(
                    f'Cannot do class-balanced random sampling if data has no '
                    f'labels. Please set sample_per_epoch=0 for test data.')

            # Save the label counts for each window sampling. Cylinders
            # whose center label is IGNORE will not be sampled
            labels = torch.LongTensor(sampling['labels'])
            counts = torch.LongTensor(sampling['label_counts'])
            valid_labels = labels != IGNORE
            labels = labels[valid_labels]
            counts = counts[valid_labels]
            self._label_counts[i, labels] = counts

        if self.is_random:
            assert self._label_counts.sum() > 0, \
                'The dataset has no sampling centers with relevant classes, ' \
                'check that your data has labels, that they follow the ' \
                'nomenclature defined for KITTI360, that your dataset uses ' \
                'enough windows and has reasonable downsampling and ' \
                'cylinder sampling resolutions.'

    @property
    def split(self):
        return self._split

    @property
    def has_labels(self):
        """Self-explanatory attribute needed for BaseDataset."""
        return self.split != 'test'

    @property
    def sample_per_epoch(self):
        """Rules the sampling mechanism for the dataset.

        When `self.sample_per_epoch > 0`, indexing the dataset produces
        random cylindrical sampling, picked so as to even-out the class
        distribution across the dataset.

        When `self.sample_per_epoch <= 0`, indexing the dataset
        addresses cylindrical samples in a deterministic fashion. The
        cylinder indices are ordered with respect to their acquisition
        window and the regular grid sampling of the centers in each
        window.
        """
        return self._sample_per_epoch

    @property
    def is_random(self):
        return self.sample_per_epoch > 0

    @property
    def radius(self):
        """The radius of cylindrical samples."""
        return self._radius

    @property
    def sample_res(self):
        """The resolution of the grid on which cylindrical samples are
        generated. The higher the sample_res, the less cylinders in the
        dataset.
        """
        return self._sample_res

    @property
    def windows(self):
        """Filenames of the dataset windows."""
        return self._WINDOWS[self.split]

    @property
    def paths(self):
        """Paths to the dataset windows data."""
        return [
            osp.join(self.processed_dir, p)
            for p in self.processed_3d_file_names]

    @property
    def sampling_paths(self):
        """Paths to the dataset windows sampling data."""
        return [
            osp.join(self.processed_dir, p)
            for p in self.processed_3d_sampling_file_names]

    @property
    def label_counts(self):
        """Count of cylindrical sampling center of each class, for each
        window. Used for class-balanced random sampling of cylinders in
        the dataset, when self.is_random==True.
        """
        return self._label_counts

    @property
    def sampling_sizes(self):
        """Number of cylindrical sampling, for each window. Used for
        deterministic sampling of cylinders in the dataset, when
        self.is_random==False.
        """
        return self._sampling_sizes

    @property
    def window_sizes(self):
        """Number of points for each pre_transformed window. This
        information will be passed in the Data objects generated by the
        dataset for the KITTI360Tracker to use when accumulating
        predictions on overlapping cylinders and any other voting
        schemes.
        """
        return self._window_sizes

    @property
    def window_raw_sizes(self):
        """Number of points for each raw window. This information will
        be passed in the Data objects generated by the dataset for the
        KITTI360Tracker to use when accumulating predictions on
        overlapping cylinders and any other voting schemes.
        """
        return self._window_raw_sizes

    @property
    def window(self):
        """Currently loaded window."""
        return self._window

    @property
    def window_idx(self):
        """Index of the currently loaded window in self.windows."""
        return self._window_idx

    @property
    def raw_file_names(self):
        """The file paths to find in order to skip the download."""
        return ['data_3d_semantics', 'data_3d_semantics_test']

    @property
    def raw_3d_paths(self):
        """These are the absolute paths to the raw window files for the
        dataset. This is used by the KITTI360Tracker to compute
        full-resolution metrics.
        """
        # The directory where train/test raw scans are
        raw_3d_dir = self.raw_file_names[1] if self.split == 'test' \
            else self.raw_file_names[0]
        return [
            osp.join(
                self.raw_dir, raw_3d_dir, x.split('/')[0], 'static',
                x.split('/')[1] + '.ply')
            for x in self.windows]

    @property
    def processed_3d_file_names(self):
        # For 'trainval', we use files from 'train' and 'val' to save
        # memory
        if self.split == 'trainval':
            return [
                osp.join(s, '3d', f'{w}.pt')
                for w in self.windows
                for s in ('train', 'val')]

        return [osp.join(self.split, '3d', f'{w}.pt') for w in self.windows]

    @property
    def processed_3d_sampling_file_names(self):
        # For 'trainval', we use files from 'train' and 'val' to save
        # memory
        if self.split == 'trainval':
            return [
                osp.join(s, '3d', f'{w}_{hash(self.sample_res)}.pt')
                for w in self.windows
                for s in ('train', 'val')]

        return [
            osp.join(self.split, '3d', f'{w}_{hash(self.sample_res)}.pt')
            for w in self.windows]

    @property
    def processed_file_names(self):
        """The name of the files to find in the :obj:`self.processed_dir`
        folder in order to skip the processing
        """
        return self.processed_3d_file_names \
               + self.processed_3d_sampling_file_names

    @property
    def submission_dir(self):
        """Submissions are saved in the `submissions` folder, in the
        same hierarchy as `raw` and `processed` directories. Each
        submission has a sub-directory of its own, named based on the
        date and time of creation.
        """
        submissions_dir = osp.join(self.root, "submissions")
        date = '-'.join([
            f'{getattr(datetime.now(), x)}'
            for x in ['year', 'month', 'day']])
        time = '-'.join([
            f'{getattr(datetime.now(), x)}'
            for x in ['hour', 'minute', 'second']])
        submission_name = f'{date}_{time}'
        path = osp.join(submissions_dir, submission_name)
        return path

    def download(self):
        raise NotImplementedError(
            'KITTI360 automatic download not implemented yet')

    def process(self):
        for path_tuple in tq(zip(self.paths, self.sampling_paths)):
            self._process_3d(*path_tuple)

    def _process_3d(self, window_path, sampling_path, return_loaded=False):
        """Internal method called by `self.process` to preprocess 3D
        points. This function is not directly written in `self.process`
        so as to help `KITTI360CylinderMM.process` benefit from this
        method to avoid re-loading the same window multiple times.
        """
        # If required files exist, skip processing
        if osp.exists(window_path) and osp.exists(sampling_path):
            if return_loaded:
                return torch.load(window_path), torch.load(sampling_path)
            else:
                return

        # Extract useful information from <path>
        split, modality, sequence_name, window_name = \
            osp.splitext(window_path)[0].split('/')[:-4]

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
            num_raw_points = data.num_nodes

            # Apply pre_transform
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # Pre-compute KD-Tree to save time when sampling later
            tree = KDTree(np.asarray(data.pos[:, :-1]), leaf_size=10)
            data[cT.CylinderSampling.KDTREE_KEY] = tree

            # Save the number of points in raw window
            data.num_raw_points = num_raw_points

            # Save pre_transformed data to the processed dir/<path>
            torch.save(data, window_path)

        else:
            data = torch.load(window_path)

        # Prepare data to build cylinder centers. Only keep 'pos'
        # and 'y' (if any) attributes and drop the z coordinate in
        # 'pos'. NB: we initialize centers as a clone of data and
        # modify centers inplace subsequently.
        centers = data.clone()
        for key in centers.keys:
            if key not in ['pos', 'y']:
                delattr(centers, key)
        centers.pos[:, 2] = 0

        # Compute the sampling of cylinder centers for the window
        sampler = cT.GridSampling3D(size=self.sample_res)
        centers = sampler(centers)
        centers.pos = centers.pos[:, :2]
        sampling = {
            'data': centers,
            'labels': None,
            'label_counts': None,
            'grid_size': self.sample_res,
            'num_points': data.num_nodes,
            'num_raw_points': data.num_raw_points}

        # If data has labels (ie not test set), save which labels
        # are present in the window and their count. These will be
        # used at sampling time to pick cylinders so as to even-out
        # class distributions
        if hasattr(centers, 'y'):
            unique, counts = np.unique(
                np.asarray(centers.y), return_counts=True)
            sampling['labels'] = unique
            sampling['label_counts'] = counts

        torch.save(sampling, sampling_path)

        if return_loaded:
            return data, sampling

    def _load_window(self, idx):
        """Load a window and its sampling data into memory based on its
        index in `self.windows`.
        """
        # Check if the window is not already loaded
        if self.window_idx == idx:
            return

        # Load the window data and associated sampling data
        self._window = Window(self.paths[idx], self.sampling_paths[idx])
        self._window_idx = idx

    def __len__(self):
        return self.sample_per_epoch if self.is_random \
            else self.sampling_sizes.sum()

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
        # Pick a 3D cylindrical sample. This will take care of 'smart'
        # window loading for us
        if self.is_random and isinstance(idx, tuple):
            data = self._get_from_label_and_window_idx(*idx)
        else:
            data = self._get_from_global_idx(idx)

        # Apply 3D transforms
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
        sampler = cT.CylinderSampling(
            self.radius, center, align_origin=False)
        data = sampler(self.window.data)

        # Save the window index and center index in the data. This will
        # be used in the KITTI360Tracker to accumulate per-window votes
        data.idx_window = int(idx_window)
        data.idx_center = int(idx_center)

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
        sampler = cT.CylinderSampling(self.radius, center, align_origin=False)
        data = sampler(self.window.data).clone()

        # Save the window index and center index in the data. This will
        # be used in the KITTI360Tracker to accumulate per-window votes
        data.idx_window = int(idx_window)
        data.idx_center = int(idx_center)

        return data

    def _pick_random_label_and_window(self):
        """Generate a `(label, idx_window)` tuple as expected by
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
#                         MiniKITTI360Cylinder                         #
########################################################################

class MiniKITTI360Cylinder(KITTI360Cylinder):
    """A mini version of KITTI360Cylinder with only a few windows for
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
                indices.append(torch.arange(group_start, i + 1))

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

        cls = MiniKITTI360Cylinder if dataset_opt.get('mini', False) \
            else KITTI360Cylinder
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
            # TODO: find an elegant way of returning class weights for
            #  train set
            raise NotImplementedError(
                'KITTI360Dataset does not support class weights yet.')

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
