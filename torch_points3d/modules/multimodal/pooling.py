from abc import ABC

import torch
import torch.nn as nn
from torch_scatter import segment_csr, scatter_min, scatter_max
from torch_points3d.core.common_modules import Seq
import math


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

    def __init__(self, mode='max', save_last=False, **kwargs):
        super(BimodalCSRPool, self).__init__()
        assert mode in self._POOLING_MODES, \
            f"Unsupported mode '{mode}'. Expected one of: {self._POOLING_MODES}"
        self._mode = mode
        self.save_last = save_last

    def forward(self, x_main, x_mod, x_proj, csr_idx):
        # Segment_CSR is "the fastest method to apply for grouped
        # reductions."
        x_pool = segment_csr(x_mod, csr_idx, reduce=self._mode)
        x_seen = csr_idx[1:] > csr_idx[:-1]
        if self.save_last:
            self._last_x_proj = x_proj
            self._last_x_mod = x_mod
            self._last_idx = torch.arange(csr_idx.shape[0] - 1, device=x_mod.device
                ).repeat_interleave(csr_idx[1:] - csr_idx[:-1])
            self._last_view_num = csr_idx[1:] - csr_idx[:-1]
        return x_pool, x_seen


class HeuristicBimodalCSRPool(nn.Module, ABC):
    """Bimodal pooling modules select and combine information from a
    modality to prepare its fusion into the main modality.

    The modality pooling may typically be used for atomic-level
    aggregation or view-level aggregation. To illustrate, in the case of
    image modality, where each 3D point may be mapped to multiple
    pixels, in multiple images. The atomic-level corresponds to pixel-
    level information, while view-level accounts for multi-image views.

    HeuristicBimodalCSRPool selects modality features based on a
    handcrafted heuristic on mapping features.

    For computation speed reasons, the data and pooling indices are
    expected to be provided in a CSR format, where same-index rows are
    consecutive and indices hold index-change pointers.

    IMPORTANT: the order of 3D points in the main modality is expected
    to match that of the indices in the mappings. Any update of the
    mappings following a reindexing, reordering or sampling of the 3D
    points must be performed prior to the multimodal pooling.
    """

    _MODES = ['max', 'min']
    _FEATURES = [
        'normalized_depth',
        'linearity',
        'planarity',
        'scattering',
        'orientation_to_the_surface',
        'normalized_pixel_height',
        'density',
        'occlusion']

    def __init__(self, mode='max', feat=0, save_last=False, **kwargs):
        super(HeuristicBimodalCSRPool, self).__init__()

        assert mode in self._MODES, \
            f"Unsupported mode '{mode}'. Expected one of: {self._MODES}."
        self._mode = mode
        self._scatter = scatter_max if mode == 'max' else scatter_min

        feat = self._FEATURES.index(feat) if isinstance(feat, str) else feat
        assert feat < len(self._FEATURES), \
            f"Feat={feat} is too large. Expected feat<{len(self._FEATURES)}."
        self._feat = feat

        self.save_last = save_last

    def forward(self, x_main, x_mod, x_proj, csr_idx):
        # Compute dense indices from CSR indices
        n_groups = csr_idx.shape[0] - 1
        dense_idx = torch.arange(n_groups, device=x_mod.device
            ).repeat_interleave(csr_idx[1:] - csr_idx[:-1])

        # Compute the arguments for the min/max heuristic
        # NB: arg_idx will carry '-1' or 'n_points' for unseen points
        _, arg_idx = self._scatter(x_proj[:, self._feat], dense_idx, dim_size=n_groups)

        # Pool the modality features based on the heuristic
        # NB: append '0' to x_mod to distribute '0' to unseen points
        x_mod_0 = torch.cat((x_mod, torch.zeros_like(x_mod[[0]])))
        x_pool = x_mod_0[arg_idx]

        x_seen = csr_idx[1:] > csr_idx[:-1]

        if self.save_last:
            self._last_x_proj = x_proj
            self._last_x_mod = x_mod
            self._last_idx = torch.arange(csr_idx.shape[0] - 1, device=x_mod.device
                ).repeat_interleave(csr_idx[1:] - csr_idx[:-1])
            self._last_view_num = csr_idx[1:] - csr_idx[:-1]
        return x_pool, x_seen

    def extra_repr(self) -> str:
        return f'mode={self._mode}, feat={self._FEATURES[self._feat]}'


class GroupBimodalCSRPool(nn.Module, ABC):
    """Bimodal pooling modules select and combine information from a
    modality to prepare its fusion into the main modality.

    The modality pooling may typically be used for atomic-level
    aggregation or view-level aggregation. To illustrate, in the case of
    image modality, where each 3D point may be mapped to multiple
    pixels, in multiple images. The atomic-level corresponds to pixel-
    level information, while view-level accounts for multi-image views.

    GroupBimodalCSRPool learns to produce a weighting scheme for the
    modality features of the same index-group, only based on the
    mapping features, optionally from the modality features
    themselves and an optional gating mechanism. This differs from the
    Key-Query attention mechanism in the sense that the main modality's
    features are not used to compute a compatibility score here.

    For computation speed reasons, the data and pooling indices are
    expected to be provided in a CSR format, where same-index rows are
    consecutive and indices hold index-change pointers.

    IMPORTANT: the order of 3D points in the main modality is expected
    to match that of the indices in the mappings. Any update of the
    mappings following a reindexing, reordering or sampling of the 3D
    points must be performed prior to the multimodal pooling.

    Example:

    import torch
    from torch_points3d.modules.multimodal.pooling import GroupBimodalCSRPool

    N = 5
    V = 20
    F_main = 10
    F_mod = 7
    F_proj = 3
    num_groups = 2

    csr_idx = torch.LongTensor([0, 4, 4, 5, 10, 20])
    x_main = None
    x_mod = torch.rand(V, F_mod)
    x_proj = torch.rand(V, F_proj)

    module = GroupBimodalCSRPool(in_proj=F_proj, in_mod=F_mod,
         num_groups=num_groups, use_mod=False, proj_min=False,
         proj_max=False, proj_num=False, gating=True,
         group_scaling=True)

    module(x_main, x_mod, x_proj, csr_idx)
    """

    def __init__(self, in_proj=None, in_mod=None, num_groups=None,
                 use_mod=False, proj_min=False, proj_max=False,
                 proj_num=False, gating=True, group_scaling=False,
                 save_last=False, **kwargs):
        super(GroupBimodalCSRPool, self).__init__()

        self.save_last = save_last

        # Group and channel arguments
        assert 1 <= num_groups <= in_mod,\
            f"Number of groups must be between 1 and in_mod={in_mod}."
        self.num_groups = num_groups
        self.in_mod = in_mod
        self.use_mod = use_mod

        # Optional computation mechanisms
        self.proj_min = proj_min
        self.proj_max = proj_max
        self.proj_num = proj_num

        # Optional compatibilities scaling mechanism
        self.group_scaling = group_scaling

        # Raw handcrafted mapping features are fed to this module, we
        # let the network preprocess them to its liking before mixing
        # them with other learnt features
        in_proj_0 = in_proj * (1 + proj_min + proj_max) + proj_num
        # in_proj_1 = nearest_power_of_2((in_proj_0 + num_groups) / 2, min_power=8)
        # in_proj_2 = nearest_power_of_2(num_groups, min_power=4)
        # in_proj_1 = nearest_power_of_2((in_proj_0 + num_groups) / 2, min_power=8)
        # in_proj_2 = nearest_power_of_2(num_groups, min_power=4)
        # self.MLP_proj = (
        #     Seq().append(nn.Linear(in_proj_0, in_proj_1, bias=True))
        #         .append(nn.BatchNorm1d(in_proj_1))
        #         .append(nn.ReLU())
        #         .append(nn.Linear(in_proj_1, in_proj_2, bias=True))
        #         .append(nn.BatchNorm1d(in_proj_2))
        #         .append(nn.ReLU()))
        # TODO: add dropout in here ?
        in_proj_2 = 4
        self.MLP_proj = (
            Seq().append(nn.Linear(in_proj_0, in_proj_2, bias=True))
                .append(nn.BatchNorm1d(in_proj_2))
                .append(nn.ReLU()))

        # Attention scores computation module
        if self.use_mod:
            in_proj_3 = in_proj_2 + in_mod
        else:
            in_proj_3 = in_proj_2
        # Ease-in the dimensionality reduction to num_groups if need be
        if num_groups <= 4 and in_proj_3 > 16:
            in_proj_4 = 8
            self.MLP_score = (
                Seq().append(nn.Linear(in_proj_3, in_proj_4, bias=True))
                    .append(nn.BatchNorm1d(in_proj_4))
                    .append(nn.ReLU())
                    .append(nn.Linear(in_proj_4, num_groups, bias=True)))
        else:
           self.MLP_score = nn.Linear(in_proj_3, num_groups, bias=True)

        # Optional gating mechanism
        self.Gating = Gating(num_groups, bias=True) if gating else None

    def forward(self, x_main, x_mod, x_proj, csr_idx):
        """
        :param x_main: N x F_main
        :param x_mod: V x F_mod
        :param x_proj: V x F_proj
        :param csr_idx:
        :return: x_pool, x_seen
        """

        # Optionally expand x_proj with difference-to-min or
        # difference-to-max or group size features
        if self.proj_min:
            x_proj_min = x_proj - segment_csr_gather(x_proj, csr_idx, reduce='min')
        else:
            x_proj_min = torch.empty(0, device=x_proj.device)
        if self.proj_max:
            x_proj_max = x_proj - segment_csr_gather(x_proj, csr_idx, reduce='max')
        else:
            x_proj_max = torch.empty(0, device=x_proj.device)
        if self.proj_num:
            # Heuristic to normalize in [0,1]
            x_proj_num = torch.sqrt(1 / (csr_idx[1:] - csr_idx[:-1]).float()
                ).repeat_interleave(csr_idx[1:] - csr_idx[:-1]).view(-1, 1)
        else:
            x_proj_num = torch.empty(0, device=x_proj.device)
        x_proj = torch.cat([x_proj, x_proj_min, x_proj_max, x_proj_num], dim=1)

        # Compute compatibilities (unscaled scores) : V x num_groups
        if self.use_mod:
            C = self.MLP_score(torch.cat([self.MLP_proj(x_proj), x_mod], dim=1))
        else:
            C = self.MLP_score(self.MLP_proj(x_proj))

        # Compute attention scores : V x num_groups
        A = segment_csr_softmax(C, csr_idx, scaling=self.group_scaling)

        # Apply attention scores : P x F_mod
        x_pool = segment_csr(
            x_mod * expand_group_feat(A, self.num_groups, self.in_mod),
            csr_idx, reduce='sum')

        if self.Gating:
            # Compute pointwise gating : P x num_groups
            G = self.Gating(segment_csr(C, csr_idx, reduce='max'))

            # Apply gating to the features : P x F_mod
            x_pool = x_pool * expand_group_feat(G, self.num_groups, self.in_mod)

        # Compute the boolean mask of seen points
        x_seen = csr_idx[1:] > csr_idx[:-1]

        # Optionally save outputs
        if self.save_last:
            self._last_x_proj = x_proj
            self._last_x_mod = x_mod
            self._last_idx = torch.arange(
                csr_idx.shape[0] - 1, device=x_proj.device
                ).repeat_interleave(csr_idx[1:] - csr_idx[:-1])
            self._last_view_num = csr_idx[1:] - csr_idx[:-1]
            self._last_C = expand_group_feat(C, self.num_groups, self.in_mod)
            self._last_A = expand_group_feat(A, self.num_groups, self.in_mod)
            if self.Gating:
                self._last_G = expand_group_feat(G, self.num_groups, self.in_mod)

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
    on mapping features, the main modality features and an optional
    gating mechanism.

    For computation speed reasons, the data and pooling indices are
    expected to be provided in a CSR format, where same-index rows are
    consecutive and indices hold index-change pointers.

    IMPORTANT: the order of 3D points in the main modality is expected
    to match that of the indices in the mappings. Any update of the
    mappings following a reindexing, reordering or sampling of the 3D
    points must be performed prior to the multimodal pooling.

    Example:

    import torch
    from torch_points3d.modules.multimodal.pooling import AttentiveBimodalCSRPool

    N = 5
    V = 20
    F_main = 10
    F_mod = 7
    F_proj = 3
    in_score = 2

    csr_idx = torch.LongTensor([0, 4, 4, 5, 10, 20])
    x_main = torch.rand(N, F_main)
    x_mod = torch.rand(V, F_mod)
    x_proj = torch.rand(V, F_proj)

    module = AttentiveBimodalCSRPool(in_main=F_main, in_proj=F_proj,
        in_mod=F_mod, in_score=in_score, proj_min=False, proj_max=False,
        proj_num=False, gating=True, dim_scaling=True,
        group_scaling=True)

    module(x_main, x_mod, x_proj, csr_idx)
    """

    def __init__(self, in_main=None, in_proj=None, in_mod=None,
                 in_score=None, proj_min=False, proj_max=False,
                 proj_num=False, gating=True, dim_scaling=True,
                 group_scaling=False, debug=False, save_last=False,
                 **kwargs):
        super(AttentiveBimodalCSRPool, self).__init__()

        self.save_last = save_last
        self.debug = debug
        if debug:
            proj_min = False
            proj_max = False
            proj_num = False
            group_scaling = False
            dim_scaling = True
            in_score = 1
            in_proj = 1
            in_mod = None

        # Optional key computation mechanisms
        self.proj_min = proj_min
        self.proj_max = proj_max
        self.proj_num = proj_num

        # Optional compatibilities scaling mechanism
        self.dim_scaling = dim_scaling
        self.group_scaling = group_scaling

        # Queries computation module
        self.Q = nn.Linear(in_main, in_score, bias=True)

        # Raw handcrafted mapping features are fed to this module, we
        # let the network preprocess them to its liking before computing
        # the keys
        in_proj_0 = in_proj * (1 + proj_min + proj_max) + proj_num
        in_proj_1 = nearest_power_of_2((in_proj_0 + in_score) / 2, min_power=16)
        in_proj_2 = nearest_power_of_2(in_score, min_power=16)
        if self.debug:
            in_proj_1 = 2
            in_proj_2 = 2
        self.MLP_proj = (
            Seq().append(nn.Linear(in_proj_0, in_proj_1, bias=True))
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

        # Optional gating mechanism
        self.Gating = Gating(1, bias=True) if gating else None

    def forward(self, x_main, x_mod, x_proj, csr_idx):
        """
        :param x_main: N x F_main
        :param x_mod: V x F_mod
        :param x_proj: V x F_proj
        :param csr_idx:
        :return: x_pool, x_seen
        """
        # Artificial x_proj and x_mod
        if self.debug:
            device = x_proj.device
            x_proj = torch.rand((x_proj.shape[0], 1), device=device)
            idx_destroyed = torch.where(x_proj < 0.3)[0]
            x_mod[idx_destroyed] = torch.rand((idx_destroyed.shape[0], *(x_mod.shape[1:])), device=device)

        # Optionally expand x_proj with difference-to-min or
        # difference-to-max or group size features
        if self.proj_min:
            x_proj_min = x_proj - segment_csr_gather(x_proj, csr_idx, reduce='min')
        else:
            x_proj_min = torch.empty(0, device=x_proj.device)
        if self.proj_max:
            x_proj_max = x_proj - segment_csr_gather(x_proj, csr_idx, reduce='max')
        else:
            x_proj_max = torch.empty(0, device=x_proj.device)
        if self.proj_num:
            # Heuristic to normalize in [0,1]
            x_proj_num = torch.sqrt(1 / (csr_idx[1:] - csr_idx[:-1]).float()
                ).repeat_interleave(csr_idx[1:] - csr_idx[:-1]).view(-1, 1)
        else:
            x_proj_num = torch.empty(0, device=x_proj.device)
        x_proj = torch.cat([x_proj, x_proj_min, x_proj_max, x_proj_num], dim=1)

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
        C = (K * Q).sum(dim=1)

        # Optionally scale compatibilities by the number of key features
        if self.dim_scaling:
            C = C / math.sqrt(K.shape[1])

        # Compute attentions : V
        A = segment_csr_softmax(C, csr_idx, scaling=self.group_scaling)

        # Compute attention-weighted modality features : P x F_mod
        x_pool = segment_csr(x_mod * A.view(-1, 1), csr_idx, reduce='sum')

        if self.Gating:
            # Compute pointwise gating : P
            G = self.Gating(segment_csr(C, csr_idx, reduce='max').view(-1, 1))

            # Apply gating to the features : P x F_mod
            x_pool = x_pool * G.view(-1, 1)

        # Compute the boolean mask of seen points
        x_seen = csr_idx[1:] > csr_idx[:-1]

        # Optionally save outputs
        if self.save_last:
            self._last_x_proj = x_proj
            self._last_x_mod = x_mod
            self._last_idx = torch.arange(csr_idx.shape[0] - 1, device=x_proj.device
                                          ).repeat_interleave(csr_idx[1:] - csr_idx[:-1])
            self._last_view_num = csr_idx[1:] - csr_idx[:-1]
            self._last_K = K
            self._last_Q = Q
            self._last_C = C
            self._last_A = A
            if self.Gating:
                self._last_G = G

        return x_pool, x_seen


class Gating(nn.Module):
    """Rectified-tanh gating mechanism with learnable linear correction."""
    def __init__(self, num_groups, weight=True, bias=True):
        super(Gating, self).__init__()
        self.num_groups = num_groups
        self.weight = nn.Parameter(torch.ones(1, num_groups)) if weight else None
        self.bias = nn.Parameter(torch.zeros(1, num_groups)) if bias else None

    def forward(self, x):
        if self.weight is not None:
            x = self.weight * x
        if self.bias is not None:
            x = x + self.bias
        return torch.tanh(torch.relu(x)).view(-1, self.num_groups).squeeze(1)

    def extra_repr(self) -> str:
        return f'weight={self.weight is not None}, bias={self.bias is not None}'


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


def group_sizes(num_elements, num_groups):
    """Local helper to compute the group sizes, when distributing
    num_elements across num_groups while keeping group sizes as close
    as possible."""
    sizes = torch.full((num_groups,), math.floor(num_elements / num_groups),
                       dtype=torch.long)
    sizes += torch.arange(num_groups) < num_elements - sizes.sum()
    return sizes


def expand_group_feat(A, num_groups, num_channels):
    if num_groups == 1:
        A = A.view(-1, 1)
    elif num_groups < num_channels:
        # Expand compatibilities to features of the same group
        sizes = group_sizes(num_channels, num_groups).to(A.device)
        A = A.repeat_interleave(sizes, dim=1)
    return A


@torch.jit.script
def segment_csr_softmax(src: torch.Tensor, csr_idx: torch.Tensor,
        eps: float = 1e-12, scaling: bool = False) -> torch.Tensor:
    """Equivalent of scatter_softmax but for CSR indices.
    Based on: torch_scatter/composite/softmax.py

    The `scaling` option allows for scaled softmax computation, where
    `scaling='True'` scales by the number of items in each index group.
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

    # Compute dense indices from CSR indices
    n_groups = csr_idx.shape[0] - 1
    dense_idx = torch.arange(n_groups).repeat_interleave(
        csr_idx[1:] - csr_idx[:-1])
    if src.dim() > 1:
        dense_idx = dense_idx.view(-1, 1).repeat(1, src.shape[1])

    # Center scores maxima near 1 for computation precision
    max_value_per_index = segment_csr(src, csr_idx, reduce='max')
    max_per_src_element = max_value_per_index.gather(0, dense_idx)
    centered_scores = src - max_per_src_element

    # Optionally scale scores by the sqrt of index group sizes
    if scaling:
        num_per_index = (csr_idx[1:] - csr_idx[:-1])
        sqrt_num_per_index = num_per_index.float().sqrt()
        num_per_src_element = torch.repeat_interleave(sqrt_num_per_index,
                                                      num_per_index)
        if src.dim() > 1:
            num_per_src_element = num_per_src_element.view(-1, 1).repeat(1,
                 src.shape[1])

        centered_scores = centered_scores / num_per_src_element

    # Compute the numerators
    centered_scores_exp = centered_scores.exp()

    # Compute the denominators
    sum_per_index = segment_csr(centered_scores_exp, csr_idx, reduce='sum')
    normalizing_constants = sum_per_index.add_(eps).gather(0, dense_idx)

    return centered_scores_exp.div(normalizing_constants)


@torch.jit.script
def segment_csr_gather(src: torch.Tensor, csr_idx: torch.Tensor,
        reduce: str = 'sum') -> torch.Tensor:
    """Compute the reduced value between same-index elements, for CSR
    indices, and redistribute them to input elements.
    """
    if not torch.is_floating_point(src):
        raise ValueError(
            '`segment_csr_gather` can only be computed over tensors with '
            'floating point data types.')
    if csr_idx.dim() != 1:
        raise ValueError(
            '`segment_csr_gather can only be computed over 1D CSR indices.')
    if src.dim() > 2:
        raise NotImplementedError(
            '`segment_csr_gather` can only be computed over 1D or 2D source '
            'tensors.')

    # Compute dense indices from CSR indices
    n_groups = csr_idx.shape[0] - 1
    dense_idx = torch.arange(n_groups).repeat_interleave(
        csr_idx[1:] - csr_idx[:-1])
    if src.dim() > 1:
        dense_idx = dense_idx.view(-1, 1).repeat(1, src.shape[1])

    # Center scores maxima near 1 for computation precision
    reduced_value_per_index = segment_csr(src, csr_idx, reduce=reduce)
    reduced_value_per_src_element = reduced_value_per_index.gather(0, dense_idx)
    return reduced_value_per_src_element


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
