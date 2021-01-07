import torch.nn as nn
import torch_scatter


class BimodalPool(nn.Module):
    """Bimodal pooling modules select and combine information from a
    modality to prepare its fusion into the main modality.

    The modality pooling may encompass two steps: a unit-level aggregation and
    a view-level aggregation. To illustrate, in the case of image modality,
    where each 3D point may be mapped to multiple pixels, in multiple images.
    The unit-level corresponds to pixel-level information, while view-level
    accounts for multi-image views.

    Various types of pooling are supported at each step: max, min, mean,
    attention-based...

    IMPORTANT: the order of 3D points in the main modality is expected to
    match that of the indexes in the mappings. Any update of the mappings
    following a reindexing, reordering or sampling of the 3D points must be
    performed prior to the multimodal pooling.
    """

    def __init__(self, unit_pool='max', view_pool='max', **kwargs):
        super().__init__()
        self.unit_pool = BimodalCSRPool(unit_pool)
        self.view_pool = BimodalCSRPool(view_pool)

    def forward(self, x_main, x_mod, mappings):
        # Unit-level aggregation
        # TODO mappings provide unit-level idx. Must not have gradients
        csr_idx = mappings.get_csr_idx(level='unit')
        x_agg = self.unit_pool(x_main, x_mod, csr_idx)

        # View-level aggregation
        # TODO mappings provide view-level idx.
        #  WARNING: will the idx order still be OK after the unit-pooling ?
        csr_idx = mappings.get_csr_idx(level='view')
        x_agg = self.view_pool(x_main, x_agg, csr_idx)

        return x_agg


class BimodalCSRPool(nn.Module):
    def __init__(self, mode='max'):
        super().__init__()
        if mode in ['max', 'mean', 'min', 'sum']:
            # TODO: beware of empty group index after torch scatter !
            self.pool = lambda x_main, x_mod, csr_idx: torch_scatter.segment_csr(x_mod, csr_idx, reduce=mode)
        else:
            # TODO create the attention-based pooling (see notes below for softmax on CSR)
            raise NotImplementedError

    def forward(self, x_main, x_mod, csr_idx):
        return self.pool(x_main, x_mod, csr_idx)

"""
# EXAMPLES

import torch
import torch_scatter

# torch_scatter.segment_coo(reduce='max')
# torch_scatter.segment_csr(X, jumps, reduce='max')
# torch_scatter.segment_csr(X, jumps, reduce='min')
# torch_scatter.segment_csr(X, jumps, reduce='mean')
# torch_scatter.segment_csr(X, jumps, reduce='sum')
# REMARK : 0-size groups will appear in the output with 0 values. So unseen points will receive zero.

n_groups = 10
jumps = torch.cumsum(torch.randint(low=0, high=3, size=(n_groups+1,)), 0)
jumps = jumps - jumps[0]
idx = torch.repeat_interleave(torch.arange(jumps.shape[0] - 1), jumps[1:] - jumps[:-1])

n_points = jumps[-1]
src = torch.randint(low=0, high=20, size=(n_points, 3))

jumps = jumps.cuda()
idx = idx.cuda()
src = src.cuda()

# CSR - jump indices
# Due to the use of index pointers, segment_csr() is the fastest method to apply for grouped reductions.
# In contrast to scatter() and segment_coo(), this operation is fully-deterministic."
torch_scatter.segment_csr(src, jumps, reduce='sum')

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
# torch_scatter.segment_csr(src, jumps, reduce='max')

max_value_per_index.gather(dim, idx.reshape((-1,1)))



"""