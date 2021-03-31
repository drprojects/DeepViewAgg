import logging
from abc import ABC

import torch
import torch.nn.functional as F
import torch.nn as nn

from torch_points3d.models.base_model import BaseModel
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.applications.multimodal.no3d import No3DEncoder

# from sklearn.neighbors import NearestNeighbors
from pykeops.torch import LazyTensor

log = logging.getLogger(__name__)


class No3D(BaseModel, ABC):

    def __init__(self, option, model_type, dataset, modules):
        if not hasattr(self, '_HAS_HEAD'):
            raise NotImplementedError

        # BaseModel init
        super().__init__(option)

        # UnwrappedUnetBasedModel init
        self.backbone = No3DEncoder(option, model_type, dataset, modules)
        self._modalities = self.backbone._modalities

        # Segmentation head init
        if self._HAS_HEAD:
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
        if self._HAS_HEAD:
            logits = self.head(features)
        else:
            logits = features
        self.output = F.log_softmax(logits, dim=-1)

        if not self.training:
            # If the module is in eval mode, propagate the output of the
            # nearest seen point to unseen points
#             nn_search = NearestNeighbors(
#                 n_neighbors=1, algorithm="kd_tree").fit(
#                 data.pos[seen_mask].detach().cpu().numpy())
#             _, nn_idx = nn_search.kneighbors(
#                 data.pos[~seen_mask].detach().cpu().numpy())
#             nn_idx = torch.LongTensor(nn_idx)
            
            # If the module is in eval mode, propagate the output of the
            # nearest seen point to unseen points
            # K-NN search with KeOps
            xyz_query_keops = LazyTensor(data.pos[~seen_mask][:, None, :])
            xyz_search_keops = LazyTensor(data.pos[seen_mask][None, :, :])
            d_keops = ((xyz_query_keops - xyz_search_keops) ** 2).sum(dim=2)
            nn_idx = d_keops.argmin(dim=1)
            del xyz_query_keops, xyz_search_keops, d_keops
            
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


class No3DFeatureFusion(No3D):
    _HAS_HEAD = True


class No3DLogitFusion(No3D):
    _HAS_HEAD = False
