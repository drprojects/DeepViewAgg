from abc import ABC

import torch
import torch.nn as nn
from torch_scatter import segment_csr
from torch_points3d.core.common_modules import Seq


class BimodalCSRPool(nn.Module, ABC):
    """Bimodal pooling modules select and combine information from a
    modality to prepare its fusion into the main modality.

    The modality pooling may typically be used for atomic-level
    aggregation or view-level aggregation. To illustrate, in the case of
    image modality, where each 3D point may be mapped to multiple
    pixels, in multiple images. The atomic-level corresponds to pixel-
    level information, while view-level accounts for multi-image views.

    BimodalCSRPool supports max, min, mean and sum pooling.

    For computation speed reasons, the data and pooling indices are
    expected to be provided in a CSR format, where same-index rows are
    consecutive and indices hold index-change pointers.

    IMPORTANT: the order of 3D points in the main modality is expected
    to match that of the indices in the mappings. Any update of the
    mappings following a reindexing, reordering or sampling of the 3D
    points must be performed prior to the multimodal pooling.
    """

    _POOLING_MODES = ['max', 'mean', 'min', 'sum']

    def __init__(self, mode='max', **kwargs):
        super(BimodalCSRPool, self).__init__()
        assert mode in self._POOLING_MODES, \
            f"Unsupported mode '{mode}'. Expected one of: {self._POOLING_MODES}"
        self._mode = mode

    def forward(self, x_main, x_mod, x_proj, csr_idx):
        # Segment_CSR is "the fastest method to apply for grouped
        # reductions."
        x_pool = segment_csr(x_mod, csr_idx, reduce=self._mode)
        x_seen = csr_idx[1:] > csr_idx[:-1]
        return x_pool, x_seen


class AttentiveBimodalCSRPool(nn.Module, ABC):
    """Bimodal pooling modules select and combine information from a
    modality to prepare its fusion into the main modality.

    The modality pooling may typically be used for atomic-level
    aggregation or view-level aggregation. To illustrate, in the case of
    image modality, where each 3D point may be mapped to multiple
    pixels, in multiple images. The atomic-level corresponds to pixel-
    level information, while view-level accounts for multi-image views.

    AttentiveBimodalCSRPool learns to attend to modality features based
    on projection features, the main modality features and an optional
    gating mechanism.

    For computation speed reasons, the data and pooling indices are
    expected to be provided in a CSR format, where same-index rows are
    consecutive and indices hold index-change pointers.

    IMPORTANT: the order of 3D points in the main modality is expected
    to match that of the indices in the mappings. Any update of the
    mappings following a reindexing, reordering or sampling of the 3D
    points must be performed prior to the multimodal pooling.
    """

    def __init__(self, in_main=None, in_proj=None, in_mod=None,
                 in_score=None, gating=True, **kwargs):
        super(AttentiveBimodalCSRPool, self).__init__()

        # Optional gating mechanism
        self.gating = gating

        # Queries computation module
        self.Q = nn.Linear(in_main, in_score, bias=True)

        # Raw handcrafted projection features are fed to this module, we
        # let the network preprocess them to its liking before computing
        # the keys
        in_proj_1 = nearest_power_of_2((in_proj + in_score) / 2, min_power=16)
        in_proj_2 = nearest_power_of_2(in_score, min_power=16)
        self.MLP_proj = (
            Seq().append(nn.Linear(in_proj, in_proj_1, bias=True))
                .append(nn.BatchNorm1d(in_proj_1))
                .append(nn.ReLU())
                .append(nn.Linear(in_proj_1, in_proj_2, bias=True))
                .append(nn.BatchNorm1d(in_proj_2))
                .append(nn.ReLU()))

        # Keys computation module
        self.mod_in_key = in_mod is not None
        if self.mod_in_key:
            self.K = nn.Linear(in_proj_2 + in_mod, in_score, bias=True)
        else:
            self.K = nn.Linear(in_proj_2, in_score, bias=True)

    def forward(self, x_main, x_mod, x_proj, csr_idx):
        """
        :param x_main: N x F_main
        :param x_mod: V x F_mod
        :param x_proj: V x F_proj
        :param csr_idx:
        :return: x_pool, x_seen
        """
        # Compute keys : V x D
        if self.mod_in_key:
            K = self.K(torch.cat([self.MLP_proj(x_proj), x_mod], dim=1))
        else:
            K = self.K(self.MLP_proj(x_proj))

        # Compute pointwise queries : N x D
        if isinstance(x_main, torch.Tensor):
            Q = self.Q(x_main)
        else:
            # For MinkowskiEngine and TorchSparse SparseTensors
            Q = self.Q(x_main.F)

        # Expand queries to views : V x D
        Q = torch.repeat_interleave(Q, csr_idx[1:] - csr_idx[:-1], dim=0)

        # Compute compatibility scores : V
        X = (K * Q).sum(dim=1)

        # Compute attentions : V
        A = segment_csr_softmax(X, csr_idx, scaling=True)

        # Compute attention-weighted modality features : P x F_mod
        x_pool = segment_csr(x_mod * A.view(-1, 1), csr_idx, reduce='max')

        if self.gating:
            # Compute pointwise gating : P
            G = segment_csr(X, csr_idx, reduce='max')
            G = torch.tanh(torch.relu(G))

            # Apply gating to the features : P x F_mod
            x_pool = x_pool * G.view(-1, 1)

        # Compute the boolean mask of seen points
        x_seen = csr_idx[1:] > csr_idx[:-1]

        return x_pool, x_seen


def nearest_power_of_2(x, min_power=16):
    """Local helper to find the nearest power of 2 of a given number.
    The `min_power` parameter puts a minimum threshold for the returned
    power.
    """
    if x < min_power:
        return min_power

    previous_power = 2 ** ((x - 1).bit_length() - 1)
    next_power = 2 ** (x - 1).bit_length()

    if x - previous_power < next_power - x:
        return previous_power
    else:
        return next_power


@torch.jit.script
def segment_csr_softmax(src: torch.Tensor, csr_idx: torch.Tensor,
        eps: float = 1e-12, scaling: bool = False) -> torch.Tensor:
    """Equivalent of scatter_softmax but for CSR indices.
    Based on: torch_scatter/composite/softmax.py

    The `scaling` option allows for scaled softmax computation, where
    values are scaled by the number of items in each index group.
    """
    if not torch.is_floating_point(src):
        raise ValueError(
            '`segment_csr_softmax` can only be computed over tensors with '
            'floating point data types.')
    if csr_idx.dim() != 1:
        raise ValueError(
            '`segment_csr_softmax` can only be computed over 1D CSR indices.')
    if src.dim() > 2:
        raise NotImplementedError(
            '`segment_csr_softmax` can only be computed over 1D or 2D source '
            'tensors.')

    n_groups = csr_idx.shape[0] - 1

    dense_idx = torch.arange(n_groups).repeat_interleave(
        csr_idx[1:] - csr_idx[:-1])
    if src.dim() > 1:
        dense_idx = dense_idx.view(-1, 1).repeat(1, src.shape[1])

    max_value_per_index = segment_csr(src, csr_idx, reduce='max')
    max_per_src_element = max_value_per_index.gather(0, dense_idx)

    centered_scores = src - max_per_src_element

    if scaling:
        num_per_index = (csr_idx[1:] - csr_idx[:-1])
        num_per_src_element = torch.repeat_interleave(num_per_index.float().sqrt(),
                                                      num_per_index)
        if src.dim() > 1:
            num_per_src_element = num_per_src_element.view(-1, 1).repeat(1,
                 src.shape[1])

        centered_scores = centered_scores / num_per_src_element

    centered_scores_exp = centered_scores.exp()

    sum_per_index = segment_csr(centered_scores_exp, csr_idx, reduce='sum')
    normalizing_constants = sum_per_index.add_(eps).gather(0, dense_idx)

    return centered_scores_exp.div(normalizing_constants)


"""
# EXAMPLES

import torch
import torch_scatter

# torch_scatter.segment_coo(reduce='max')
# torch_scatter.segment_csr(X, pointers, reduce='max')
# torch_scatter.segment_csr(X, pointers, reduce='min')
# torch_scatter.segment_csr(X, pointers, reduce='mean')
# torch_scatter.segment_csr(X, pointers, reduce='sum')
# REMARK : 0-size groups will appear in the output with 0 values. So unseen points will receive zero.

n_groups = 10
pointers = torch.cumsum(torch.randint(low=0, high=3, size=(n_groups+1,)), 0)
pointers = pointers - pointers[0]
idx = torch.repeat_interleave(torch.arange(pointers.shape[0] - 1), pointers[1:] - pointers[:-1])

n_points = pointers[-1]
src = torch.randint(low=0, high=20, size=(n_points, 3))

pointers = pointers.cuda()
idx = idx.cuda()
src = src.cuda()

# CSR - pointer indices
# Due to the use of index pointers, segment_csr() is the fastest method to apply for grouped reductions.
# In contrast to scatter() and segment_coo(), this operation is fully-deterministic."
torch_scatter.segment_csr(src, pointers, reduce='sum')

# COO - sorted indices
# In contrast to scatter(), this method expects values in index to be sorted along dimension index.dim() - 1.
# Due to the use of sorted indices, segment_coo() is usually faster than the more general scatter() operation.
torch_scatter.segment_coo(src, idx, reduce='sum')

#----------------------------------------------------------------------

# For attention mechanism
# on idx, not COO, nor CSR...
torch_scatter.composite.scatter_softmax

#----------------------------------------------------------------------

# To extend Softmax to CSR format:
# https://pytorch-scatter.readthedocs.io/en/1.4.0/_modules/torch_scatter/composite/softmax.html#scatter_softmax
from torch_scatter import scatter_max
dim = 0
max_value_per_index, _ = scatter_max(src, idx, dim=dim)
# same as 
# torch_scatter.segment_csr(src, pointers, reduce='max')

max_value_per_index.gather(dim, idx.reshape((-1,1)))
"""

"""
import torch
from torch_points3d.modules.multimodal.pooling import segment_csr_softmax
src = torch.arange(15).float().view(-1, 1).repeat_interleave(2, dim=1)
csr_idx = torch.LongTensor([0, 5, 10, 15])
segment_csr_softmax(src, csr_idx)

segment_csr_softmax(src, csr_idx, scaling=True)
"""
