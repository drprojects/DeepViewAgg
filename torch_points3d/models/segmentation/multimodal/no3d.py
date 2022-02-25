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

    _MODALITY_VIEW_LOSS = None

    def __init__(self, option, model_type, dataset, modules):
        # No3D should not be directly instantiated, child classes should
        # be used instead
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

        # Control the loss mechanism with MODALITY_VIEW_LOSS. If set to
        # None, the model will be trained by backpropagating the 3D
        # pointwise loss through the modality branches. If
        # MODALITY_VIEW_LOSS is set to an existing modality name, then
        # the loss used will the loss computed on the provided
        # modality's latest view-level output. This mode requires
        # setting 'keep_last_view=True' for the corresponding
        # UniModalBranch.
        if self._MODALITY_VIEW_LOSS is not None:
            assert self._MODALITY_VIEW_LOSS in self._modalities, \
                f"Cannot set modality loss for '{self._MODALITY_VIEW_LOSS}'. " \
                f"Expected one of {self._modalities}."
            # TODO: check that the corresponding UniModalBranches have
            #  'keep_last_view=True' and that the pooling mechanism does
            #  not require any learning: ie not AttentiveBimodalCSRPool.
            #  Indeed, the current view-loss training is coded in such a
            #  way that the image encoder is supervised for each view as
            #  if it was alone. If the desired view-pooling mechanism
            #  requires any training, then it will never receive any
            #  gradient from the view loss and hence will not be trained.

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
        
        if features.device != self.device:
            features = features.to(self.device)
        
        if seen_mask is None:
            seen_mask = torch.zeros(features.shape[0], dtype=torch.bool, device=self.device)

        # TODO: this is a bit dirty, saving the modality-wise feature
        #  maps in self.input. Need to find a cleaner way. This impacts
        #  the Dash visualization app.
        for m in self.modalities:
            for i in range(self.input.modalities[m].num_settings):
                if self._HAS_HEAD:
                    f_map = data[m][i].x
                    s = f_map.shape
                    f_map = f_map.permute(1, 0, 2, 3).reshape(s[1], -1).T
                    f_map = self.head(f_map)
                    f_map = f_map.T.reshape(f_map.shape[1], s[0], s[2], s[3]).permute(1, 0, 2, 3)
                    self.input.modalities[m][i].pred = f_map
                    self.input.modalities[m][i].feat = data[m][i].x
                else:
                    self.input.modalities[m][i].pred = data[m][i].x

        logits = self.head(features) if self._HAS_HEAD else features
        self.output = F.log_softmax(logits, dim=-1)

        if not self.training:
            if torch.any(seen_mask):
                # If the module is in eval mode, propagate the output of the
                # nearest seen point to unseen points
                # nn_search = NearestNeighbors(
                #     n_neighbors=1, algorithm="kd_tree").fit(
                #     data.pos[seen_mask].detach().cpu().numpy())
                # _, nn_idx = nn_search.kneighbors(
                #     data.pos[~seen_mask].detach().cpu().numpy())
                # nn_idx = torch.LongTensor(nn_idx)

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
                # If no points were seen, do not compute the loss on 
                # unseen data points
                self.labels[~seen_mask] = IGNORE_LABEL

        else:
            # If the module is in training mode, do not compute the loss
            # on the unseen data points
            self.labels[~seen_mask] = IGNORE_LABEL

        # Compute the segmentation loss
        if self.labels is not None:

            # Based on the 3D pointwise predictions
            if self._MODALITY_VIEW_LOSS is None:
                pred = self.output
                target = self.labels

            # Based on a modality's view-wise predictions
            else:
                view_features = data[self._MODALITY_VIEW_LOSS].last_view_x_mod
                csr_idx = data[self._MODALITY_VIEW_LOSS].last_view_csr_idx
                view_logits = self.head(view_features) if self._HAS_HEAD \
                    else view_features
                pred = F.log_softmax(view_logits, dim=-1)
                target = torch.repeat_interleave(self.labels,
                    csr_idx[1:] - csr_idx[:-1])

            self.loss_seg = F.nll_loss(pred, target, ignore_index=IGNORE_LABEL)

    def backward(self):
        self.loss_seg.backward()


class No3DFeatureFusion(No3D):
    _HAS_HEAD = True


class No3DLogitFusion(No3D):
    _HAS_HEAD = False


class No3DImageFeatureFusion(No3D):
    _HAS_HEAD = True
    _MODALITY_VIEW_LOSS = 'image'


class No3DImageLogitFusion(No3D):
    _HAS_HEAD = False
    _MODALITY_VIEW_LOSS = 'image'
