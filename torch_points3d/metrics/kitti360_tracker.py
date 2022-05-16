from typing import Dict, Any
import logging
import torch
import os
import os.path as osp
import glob
import tempfile
import numpy as np
from tqdm.auto import tqdm as tq
from zipfile import ZipFile

from torch_geometric.nn.unpool import knn_interpolate
from torch_geometric.data import Batch

from torch_points3d.datasets.batch import SimpleBatch
from torch_points3d.metrics.confusion_matrix import ConfusionMatrix
from torch_points3d.metrics.segmentation_tracker import SegmentationTracker
from torch_points3d.core.data_transform import SaveOriginalPosId
from torch_points3d.models import model_interface
from torch_points3d.datasets.segmentation.kitti360 import read_kitti360_window
from torch_points3d.datasets.segmentation.kitti360_config import TRAINID2ID

log = logging.getLogger(__name__)


class KITTI360Tracker(SegmentationTracker):
    # TODO: add support for tracking 'mciou' for KITTI360 Mean Category
    #  IoU

    def reset(self, *args, **kwargs):

        # SegmentationTracker handles _confusion_matrix, _acc, _macc,
        # _miou, and _miou_per_class
        super().reset(*args, **kwargs)

        # KITTI360Tracker additionally handles the following metrics, to
        # track performance with voting (each 3D point may be predicted
        # from various cylindrical sampled) and at full resolution.
        self._vote_confusion_matrix = ConfusionMatrix(self._num_classes)
        self._full_confusion_matrix = ConfusionMatrix(self._num_classes)
        self._vote_miou = None
        self._full_vote_miou = None
        self._iou_per_class = {}
        self._vote_iou_per_class = {}
        self._full_vote_iou_per_class = {}

        # Attributes to manage per-window metrics
        self.windows = self.stage_dataset.windows
        self.window_raw_files = [
            osp.join(self.stage_dataset.raw_dir, x)
            for x in self.stage_dataset.raw_file_names_3d]
        self._temp_dir = None
        self._idx_window = None
        self._votes = None
        self._counts = None

        # Initialize a submission folder path based the date and time of
        # creation of the tracker. This way, in case a submission is
        # required when calling `self.finalise`, the name of the
        # submission folder will be unique. The submission directory
        # will only be created if `self.finalise(make_submission=True)`
        # is called
        self._submission_dir = self._dataset.submission_dir

    def track(
            self, model: model_interface.TrackerInterface, data=None,
            **kwargs):
        """Add current model predictions (usually the result of a batch)
        to the tracking.
        """
        super().track(model)

        # For train set, nothing to do. For val and test sets, we want
        # to be careful with KITTI360 overlapping cylinders and
        # multi-run voting. The real metrics must be computed with
        # respect to overlaps and voting schemes
        if self._stage == "train":
            return

        # Create a temporary directory in the `/tmp` directory. If no
        # such directory is found on the machine, the dataset root
        # directory (where `raw` and `processed` folders are) will be
        # used. This is where the tracker will create a file for each
        # window, storing the per-point votes. The directory will be
        # automatically deleted when the tracker is deleted or
        # `self.reset` is called
        if self._temp_dir is None:
            tmp_dir = '/tmp' if osp.exists('/tmp') else self.stage_dataset.root
            self._temp_dir = tempfile.TemporaryDirectory(dir=tmp_dir)

        # Gather input data
        data = model.get_input() if data is None else data
        data = data.data if model.is_multimodal else data

        # Recover predictions from the model and add them to data
        # attributes
        data.pred = model.get_output()

        # If the data is batched, split into its original elements.
        # Special attention must be given to the newly-added 'pred'
        # attribute which was not present in the Data list at Batch
        # creation time
        # # TODO: this is only for torch geometric Batch, but won't work
        #    for TP3D's SimpleBatch
        if isinstance(data, Batch):
            data.__slices__['pred'] = data.__slices__['pos']
            data.__cat_dims__['pred'] = 0
            data.__cumsum__['pred'] = data.__cumsum__['pos']
            data_list = data.to_data_list()
        elif isinstance(data, SimpleBatch):
            data_list = data.to_data_list()
        else:
            data_list = [data]

        # Loop over items of the batch, because some may come from
        # different windows
        for data in data_list:
            try:
                # Get window information
                idx_window = data.idx_window
                if torch.is_tensor(idx_window):
                    idx_window = idx_window.item()
            except:
                print()
                print(f'data_list : {data_list}')
                print(f'data : {data}')
                print()

            # If the tracker's currently-loaded window traking data must
            # change, save votes and counts for the previous window and
            # load those for the new one
            if idx_window != self._idx_window:
                self._save_window_tracking()
                self._load_window_tracking(idx_window)

            # Recover the point indices in the original raw cloud
            origin_ids = data[SaveOriginalPosId.KEY]
            if origin_ids is None:
                raise ValueError(
                    f'The inputs given to the model do not have a '
                    f'{SaveOriginalPosId.KEY } attribute."')
            if origin_ids.dim() == 2:
                origin_ids = origin_ids.flatten()
            if origin_ids.max() >= self._votes.shape[0]:
                raise ValueError(
                    'Origin ids are larger than the number of points in the '
                    'original point cloud.')

            # Save predictions
            # WARNING: if a point appears multiple times in origin_ids,
            # only one of its 'outputs' will be counted
            self._votes[origin_ids] += data.pred.cpu()
            self._counts[origin_ids] += 1

    def finalise(self, full_res=False, make_submission=False, **kwargs):
        # Submission only for 'test' set and if submission is required,
        # full_res is set to True
        make_submission = make_submission and self._stage == 'test'
        full_res = make_submission or full_res

        # Since saving tracked votes and prediction counts is only
        # triggered when the window changes, we need to manually
        # save the tracked data from the last batch. This is important,
        # as this would entirely drop tracking for the entirety of the
        # last window of val and test sets !
        self._save_window_tracking()

        # Compute basic metrics without taking into account voting
        # schemes with multiple predictions on the same points. If the
        # dataset has no labels, these metrics will not be computed
        if self.has_labels:
            per_class_iou = \
                self._confusion_matrix.get_intersection_union_per_class()[0]
            self._iou_per_class = {
                self._dataset.INV_OBJECT_LABEL[k]: v
                for k, v in enumerate(per_class_iou)}

        # We don't compute the voting nor full-resolution metrics on the
        # train set
        if self._stage == 'train' or not self.has_labels \
                and not make_submission:
            return

        # Compute voting and (optionally) full-resolution predictions
        # for each window
        for idx_window in tq(range(len(self.windows))):

            # Load the votes and prediction counts
            self._load_window_tracking(idx_window)

            # Select the window points that received a prediction
            has_pred = self._counts > 0
            pred = torch.argmax(self._votes[has_pred], 1).numpy()

            # Load ground truth and/or point positions from raw data. If
            # no labels are found in the PLY file, y will simply be None
            if full_res or self.has_labels:
                full_data = read_kitti360_window(
                    self.window_raw_files[idx_window], xyz=full_res, rgb=False,
                    semantic=self.has_labels, instance=False, remap=True)

            # If labels were found compute voting metrics for the window
            # low-res points. Note that points with ignored labels are
            # masked out
            if self.has_labels and full_data.y is not None:
                gt = full_data.y[has_pred].numpy()
                mask = gt != self._ignore_label
                self._vote_confusion_matrix.count_predicted_batch(
                    gt[mask], pred[mask])

            # Stop here if full-resolution metrics are not required
            if not full_res:
                continue

            # If full-resolution metrics or benchmark submission are
            # required, compute full-resolution predictions by nearest
            # neighbor interpolation
            # TODO: NN search with faster CPU or GPU methods
            full_pred = self._votes.argmax(1).numpy()
            full_pred[~has_pred] = knn_interpolate(
                self._votes[has_pred], full_data.pos[has_pred],
                full_data.pos[~has_pred], k=1).argmax(1).numpy()

            # If labels were found compute voting metrics for the window
            # low-res points. Note that points with ignored labels are
            # masked out
            if self.has_labels and full_data.y is not None:
                gt = full_data.y.numpy()
                mask = gt != self._ignore_label
                self._full_confusion_matrix.count_predicted_batch(
                    gt[mask], full_pred[mask])

            if make_submission:
                self._make_submission(idx_window, full_pred)

        # Compress submission files into a final .zip archive as
        # expected by the KITTI360 submission server
        if make_submission:
            self._zip_submission()

        # Compute the global voting metrics for low-resolution points
        cm = self._vote_confusion_matrix
        if cm.confusion_matrix is not None and cm.confusion_matrix.sum() > 0:
            self._vote_miou = cm.get_average_intersection_union() * 100
            per_class_iou = cm.get_intersection_union_per_class()[0]
            self._vote_iou_per_class = {
                self._dataset.INV_OBJECT_LABEL[k]: 100 * v
                for k, v in enumerate(per_class_iou)}

        # Compute the global voting metrics for full-resolution points
        cm = self._full_confusion_matrix
        if cm.confusion_matrix is not None and cm.confusion_matrix.sum() > 0:
            self._full_vote_miou = cm.get_average_intersection_union() * 100
            per_class_iou = cm.get_intersection_union_per_class()[0]
            self._full_vote_iou_per_class = {
                self._dataset.INV_OBJECT_LABEL[k]: 100 * v
                for k, v in enumerate(per_class_iou)}

    def _make_submission(self, idx_window, pred):
        """Prepare data for a sumbission to KITTI360 for 3D semantic
        Segmentation on the test set.

        Expected submission format is detailed here:
        https://github.com/autonomousvision/kitti360Scripts/tree/master/kitti360scripts/evaluation/semantic_3d
        """
        if not osp.exists(self._submission_dir):
            os.makedirs(self._submission_dir)

        # Make sure the prediction is a 1D Numpy array
        if len(pred.shape) != 1:
            raise ValueError(
                f'The submission predictions must be 1D Numpy vectors, '
                f'received {type(pred)} of shape {pred.shape} instead.')

        # Map TrainId labels to expected Ids
        pred_remapped = TRAINID2ID[pred].astype(np.uint8)

        # Recover sequence and window information from stage dataset's
        # windows and format those to match the expected file name:
        # {seq:0>4}_{start_frame:0>10}_{end_frame:0>10}.npy
        sequence_name, window_name = self.windows[idx_window].split('/')
        seq = sequence_name.split('_')[-2]
        start_frame, end_frame = window_name.split('_')
        filename = f'{seq:0>4}_{start_frame:0>10}_{end_frame:0>10}.npy'

        # Save the window submission
        np.save(osp.join(self._submission_dir, filename), pred_remapped)

    def _zip_submission(self):
        """This should be called once all window submission files have
        been saved using `self._make_submission`. This will zip them
        together as expected by the KITTI360 submission server.
        """
        zipObj = ZipFile(f'{self._submission_dir}.zip', 'w')
        for p in glob.glob(osp.join(self._submission_dir, '*.npy')):
            zipObj.write(p)
        zipObj.close()

    def get_metrics(self, verbose=False) -> Dict[str, Any]:
        """Return a dictionary of all metrics and losses being tracked.
        """
        # Low-resolution metrics without voting
        metrics = super().get_metrics(verbose)
        if verbose and self.has_labels:
            metrics[f'{self._stage}_iou'] = self._iou_per_class

        # Low-resolution voting metrics
        if self._vote_miou:
            metrics[f'{self._stage}_vote_miou'] = self._vote_miou

            if verbose:
                metrics[f'{self._stage}_vote_iou'] = self._vote_iou_per_class

        # Full-resolution voting metrics
        if self._full_vote_miou:
            metrics[f'{self._stage}_full_miou'] = self._full_vote_miou

            if verbose:
                metrics[f'{self._stage}_full_iou'] = \
                    self._full_vote_iou_per_class

        return metrics

    @property
    def stage_dataset(self):
        """Dataset for the Tracker's current stage"""
        return self._dataset.get_dataset(self._stage)

    @property
    def temp_dir(self):
        """Name of the TemporaryDirectory created when tracking voting
        metrics over all windows.
        """
        return None if self._temp_dir is None else self._temp_dir.name

    @property
    def has_labels(self):
        return self._dataset.has_labels(self._stage)

    def _save_window_tracking(self):
        """Save a window's votes and prediction counts to the tracker's
        temporary directory.
        """
        # Don't save anything for the tracker's initialization state
        if self._idx_window is None or self._idx_window < 0:
            return
        window = self.windows[self._idx_window]
        temp_window_path = osp.join(self.temp_dir, window + '.pt')
        os.makedirs(osp.dirname(temp_window_path), exist_ok=True)
        torch.save((self._votes, self._counts), temp_window_path)

    def _load_window_tracking(self, idx_window):
        """Load a window's votes and prediction counts from the
        tracker's temporary directory. If the window has no votes and
        counts yet, they will be initialized here.
        """
        window = self.windows[idx_window]
        window_raw_size = self.stage_dataset.window_raw_sizes[idx_window]
        temp_window_path = osp.join(self.temp_dir, window + '.pt')
        os.makedirs(osp.dirname(temp_window_path), exist_ok=True)
        if not osp.exists(temp_window_path):
            votes = torch.zeros(window_raw_size, self._num_classes).float()
            counts = torch.zeros(window_raw_size).int()
        else:
            votes, counts = torch.load(temp_window_path)
        self._idx_window = idx_window
        self._votes = votes
        self._counts = counts
