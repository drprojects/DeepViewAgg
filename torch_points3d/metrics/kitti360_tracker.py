from typing import Dict, Any
import logging
import torch
import os
import os.path as osp
import tempfile

from torch_geometric.nn.unpool import knn_interpolate

from torch_points3d.metrics.confusion_matrix import ConfusionMatrix
from torch_points3d.metrics.segmentation_tracker import SegmentationTracker
from torch_points3d.metrics.base_tracker import BaseTracker, meter_value
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.core.data_transform import SaveOriginalPosId
from torch_points3d.models import model_interface

log = logging.getLogger(__name__)


class KITTI360Tracker(SegmentationTracker):
    def reset(self, *args, **kwargs):

        # SegmentationTracker handles _confusion_matrix, _acc, _macc,
        # _miou, and _miou_per_class
        super().reset(*args, **kwargs)

        # KITTI360Tracker additionally handles the following metrics, to
        # track performance with voting (each 3D point may be predicted
        # from various cylindrical sampled) and at full resolution.
        self._full_confusion_matrix = ConfusionMatrix(self._num_classes)
        self._vote_miou = None
        self._full_vote_miou = None
        self._iou_per_class = {}
        self._vote_iou_per_class = {}
        self._full_vote_iou_per_class = {}

        # Recover the stage-dataset's windows
        stage_dataset = self._dataset.get_dataset(self._stage)
        self.windows = stage_dataset.windows
        self.window_raw_files = stage_dataset.raw_3d_file_names
        self.temp_dir = tempfile.TemporaryDirectory()


    def track(
            self, model: model_interface.TrackerInterface, full_res=False,
            data=None, **kwargs):
        """Add current model predictions (usually the result of a batch)
        to the tracking.
        """
        super().track(model)

        # For train set, nothing to do
        if self._stage == "train" or not full_res:
            return

        # For val and test sets, we want to be careful with KITTI360
        # overlapping cylinders and multi-run voting. The real metrics
        # must be computed with respect to overlaps and voting schemes

        # if self._test_area is None:
        #     self._test_area = self._dataset.test_data.clone()
        #     if self._test_area.y is None:
        #         raise ValueError("It seems that the test area data does not have labels (attribute y).")
        #     self._test_area.prediction_count = torch.zeros(self._test_area.y.shape[0], dtype=torch.int)
        #     self._test_area.votes = torch.zeros((self._test_area.y.shape[0], self._num_classes), dtype=torch.float)
        #     self._test_area.to(model.device)
        #
        # # Gather origin ids and check that it fits with the test set
        # inputs = data if data is not None else model.get_input()
        # originids = inputs[SaveOriginalPosId.KEY] if not model.is_multimodal else inputs.data[SaveOriginalPosId.KEY]
        # if originids is None:
        #     raise ValueError("The inputs given to the model do not have a %s attribute." % SaveOriginalPosId.KEY)
        # if originids.dim() == 2:
        #     originids = originids.flatten()
        # if originids.max() >= self._test_area.pos.shape[0]:
        #     raise ValueError("Origin ids are larger than the number of points in the original point cloud.")
        #
        # # Set predictions
        # # WARNING. If a point appears multiple times in originids, only
        # # one of its 'outputs' and one 'prediction_count' will be counted
        # outputs = model.get_output()
        # self._test_area.votes[originids] += outputs
        # self._test_area.prediction_count[originids] += 1

    def finalise(self, full_res=False, vote_miou=True, **kwargs):
        per_class_iou = self._confusion_matrix.get_intersection_union_per_class()[0]
        self._iou_per_class = {self._dataset.INV_OBJECT_LABEL[k]: v for k, v in enumerate(per_class_iou)}

        # if not self._test_area:
        #     return
        #
        # if vote_miou:
        #     # Complete for points that have a prediction
        #     self._test_area = self._test_area.to("cpu")
        #     has_prediction = self._test_area.prediction_count > 0
        #     gt = self._test_area.y[has_prediction].numpy()
        #     c = ConfusionMatrix(self._num_classes)
        #     pred = torch.argmax(self._test_area.votes[has_prediction], 1).numpy()
        #     c.count_predicted_batch(gt, pred)
        #     self._vote_miou = c.get_average_intersection_union() * 100
        #     per_class_iou = c.get_intersection_union_per_class()[0]
        #     self._vote_iou_per_class = {
        #         self._dataset.INV_OBJECT_LABEL[k]: 100 * v
        #         for k, v in enumerate(per_class_iou)}
        #
        # if full_res:
        #     self._compute_full_miou()


    # def _compute_full_miou(self):
    #     if self._full_vote_miou is not None:
    #         return
    #
    #     has_prediction = self._test_area.prediction_count > 0
    #     log.info(
    #         "Computing full res mIoU, we have predictions for %.2f%% of the points."
    #         % (torch.sum(has_prediction) / (1.0 * has_prediction.shape[0]) * 100)
    #     )
    #
    #     self._test_area = self._test_area.to("cpu")
    #
    #     # Full res interpolation
    #     full_pred = knn_interpolate(
    #         self._test_area.votes[has_prediction],
    #         self._test_area.pos[has_prediction], self._test_area.pos, k=1,)
    #
    #     # Full res pred
    #     self._full_confusion = ConfusionMatrix(self._num_classes)
    #     self._full_confusion.count_predicted_batch(self._test_area.y.numpy(), torch.argmax(full_pred, 1).numpy())
    #     self._full_vote_miou = self._full_confusion.get_average_intersection_union() * 100
    #     per_class_iou = self._full_confusion.get_intersection_union_per_class()[0]
    #     self._full_vote_iou_per_class = {
    #         self._dataset.INV_OBJECT_LABEL[k]: 100 * v
    #         for k, v in enumerate(per_class_iou)}

    # @property
    # def full_confusion_matrix(self):
    #     return self._full_confusion

    def get_metrics(self, verbose=False) -> Dict[str, Any]:
        """ Returns a dictionnary of all metrics and losses being tracked
        """
        metrics = super().get_metrics(verbose)

        if verbose:
            metrics[f'{self._stage}_iou'] = self._iou_per_class

            # if self._full_vote_miou:
            #     metrics[f'{self._stage}_full_vote_miou'] = self._full_vote_miou
            #     metrics[f'{self._stage}_full_vote_iou'] = self._full_vote_iou_per_class
            #
            # if self._vote_miou:
            #     metrics[f'{self._stage}_vote_miou'] = self._vote_miou
            #     metrics[f'{self._stage}_vote_iou'] = self._vote_iou_per_class

        return metrics
