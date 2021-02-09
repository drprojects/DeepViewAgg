import logging
from abc import ABC

import torch
import torch.nn.functional as F
import torch.nn as nn

from torch_points3d.models.base_model import BaseModel
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.applications.multimodal.no3d import No3DEncoder

from sklearn.neighbors import NearestNeighbors

log = logging.getLogger(__name__)


class No3D(BaseModel, ABC):
    _REQUIRES_HEAD = True

    def __init__(self, option, model_type, dataset, modules):
        # BaseModel init
        super().__init__(option)

        # UnwrappedUnetBasedModel init
        self.backbone = No3DEncoder(option, model_type, dataset, modules)

        # Segmentation head init
        if self._REQUIRES_HEAD:
            self.head = nn.Sequential(nn.Linear(self.backbone.output_nc,
                                                dataset.num_classes))
        self.loss_names = ["loss_seg"]

    def set_input(self, data, device):
        self.input = data

        if hasattr(data, 'batch') and data.batch is not None:
            self.batch_idx = data.batch.squeeze()
        else:
            self.batch_idx = None

        if hasattr(data, 'y') and data.y is not None:
            self.labels = data.y.to(self.device)
        else:
            self.labels = None

    def forward(self, *args, **kwargs):
        data = self.backbone(self.input)
        features = data.x
        seen_mask = data.seen
        if self._REQUIRES_HEAD:
            logits = self.head(features)
        else:
            logits = features
        self.output = F.log_softmax(logits, dim=-1)

        if not self.training:
            # If the module is in eval mode, propagate the output of the
            # nearest seen point to unseen points
            nn_search = NearestNeighbors(
                n_neighbors=1, algorithm="kd_tree").fit(
                data.pos[seen_mask].detach().cpu().numpy())
            _, nn_idx = nn_search.kneighbors(
                data.pos[~seen_mask].detach().cpu().numpy())
            nn_idx = torch.LongTensor(nn_idx)
            self.output[~seen_mask] = self.output[seen_mask][nn_idx].squeeze()

        else:
            # If the module is in training mode, do not compute the loss
            # on the unseen data points
            self.labels[~seen_mask] = IGNORE_LABEL

        if self.labels is not None:
            self.loss_seg = F.nll_loss(self.output, self.labels,
                                       ignore_index=IGNORE_LABEL)

    def backward(self):
        self.loss_seg.backward()


class No3DWithViewLogitFusion(No3D):
    _REQUIRES_HEAD = False