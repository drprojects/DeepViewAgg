from abc import ABC

import torch
import torch.nn as nn
import torch_scatter


class BimodalCSRPool(nn.Module, ABC):
    """Bimodal pooling modules select and combine information from a
    modality to prepare its fusion into the main modality.

    The modality pooling may typically be used for atomic-level
    aggregation or view-level aggregation. To illustrate, in the case of
    image modality, where each 3D point may be mapped to multiple
    pixels, in multiple images. The atomic-level corresponds to pixel-
    level information, while view-level accounts for multi-image views.

    Various types of pooling are supported: max, min, mean,
    attention-based...

    For computation speed reasons, the data and pooling indices are
    expected to be provided in a CSR format, where same-index rows are
    consecutive and indices hold index-change pointers.

    IMPORTANT: the order of 3D points in the main modality is expected
    to match that of the indices in the mappings. Any update of the
    mappings following a reindexing, reordering or sampling of the 3D
    points must be performed prior to the multimodal pooling.
    """

    def __init__(self, mode='max', **kwargs):
        super(BimodalCSRPool, self).__init__()
        self.mode = mode

    def forward(self, x_main, x_mod, csr_idx):
        if self.mode in ['max', 'mean', 'min', 'sum']:
            # Segment_CSR is "the fastest method to apply for grouped
            # reductions."
            x_pool = torch_scatter.segment_csr(x_mod, csr_idx, reduce=self.mode)
            x_seen = csr_idx[1:] > csr_idx[:-1]

        else:
            # TODO create the attention-based pooling (see notes below
            #  for softmax on CSR)
            raise NotImplementedError

        return x_pool, x_seen


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
