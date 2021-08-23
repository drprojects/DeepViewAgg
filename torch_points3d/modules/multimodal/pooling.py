from abc import ABC

import sys
import torch
import torch.nn as nn
from torch_scatter import segment_csr, scatter_min, scatter_max
from torch_points3d.core.common_modules import MLP
import math

_local_modules = sys.modules[__name__]


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

        # Optional mechanism to keep track of the outputs for debugging
        # or view-wise loss
        self.save_last = save_last
        self._last_x_map = None
        self._last_x_mod = None
        self._last_idx = None
        self._last_view_num = None

    def forward(self, x_main, x_mod, x_map, csr_idx):
        # Segment_CSR is "the fastest method to apply for grouped
        # reductions."
        x_pool = segment_csr(x_mod, csr_idx, reduce=self._mode)
        x_seen = csr_idx[1:] > csr_idx[:-1]
        if self.save_last:
            self._last_x_map = x_map
            self._last_x_mod = x_mod
            self._last_idx = torch.arange(
                csr_idx.shape[0] - 1, device=x_mod.device).repeat_interleave(
                csr_idx[1:] - csr_idx[:-1])
            self._last_view_num = csr_idx[1:] - csr_idx[:-1]
        return x_pool, x_seen

    def extra_repr(self) -> str:
        return f'mode={self._mode}, save_last={self.save_last}'


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

        # Optional mechanism to keep track of the outputs for debugging
        # or view-wise loss
        self.save_last = save_last
        self._last_x_map = None
        self._last_x_mod = None
        self._last_idx = None
        self._last_view_num = None

    def forward(self, x_main, x_mod, x_map, csr_idx):
        # Compute dense indices from CSR indices
        n_groups = csr_idx.shape[0] - 1
        dense_idx = torch.arange(n_groups, device=x_mod.device
            ).repeat_interleave(csr_idx[1:] - csr_idx[:-1])

        # Compute the arguments for the min/max heuristic
        # NB: arg_idx will carry '-1' or 'n_points' for unseen points
        _, arg_idx = self._scatter(
            x_map[:, self._feat], dense_idx, dim_size=n_groups)

        # Pool the modality features based on the heuristic
        # NB: append '0' to x_mod to distribute '0' to unseen points
        x_mod_0 = torch.cat((x_mod, torch.zeros_like(x_mod[[0]])))
        x_pool = x_mod_0[arg_idx]

        x_seen = csr_idx[1:] > csr_idx[:-1]

        if self.save_last:
            self._last_x_map = x_map
            self._last_x_mod = x_mod
            self._last_idx = torch.arange(
                csr_idx.shape[0] - 1, device=x_mod.device).repeat_interleave(
                csr_idx[1:] - csr_idx[:-1])
            self._last_view_num = csr_idx[1:] - csr_idx[:-1]
        return x_pool, x_seen

    def extra_repr(self) -> str:
        return f'mode={self._mode}, feat={self._FEATURES[self._feat]}, ' \
               f'save_last={self.save_last}'


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
    F_map = 3
    n_groups = 2

    csr_idx = torch.LongTensor([0, 4, 4, 5, 10, 20])
    x_main = None
    x_mod = torch.rand(V, F_mod)
    x = torch.rand(V, F_map)

    module = GroupBimodalCSRPool(in_map=F_map, in_mod=F_mod,num_groups=n_groups)

    module(x_main, x_mod, x, csr_idx)
    """

    def __init__(
            self, in_map=None, in_mod=None, num_groups=1, use_mod=False,
            gating=True, group_scaling=True, save_last=False, nc_inner=32,
            map_encoder='DeepSetFeat', **kwargs):
        super(GroupBimodalCSRPool, self).__init__()

        # Default output feature size used for embeddings
        self.nc_inner = nc_inner

        # Optional mechanism to keep track of the outputs for debugging
        # or view-wise loss
        self.save_last = save_last
        self._last_x_map = None
        self._last_x_mod = None
        self._last_idx = None
        self._last_view_num = None
        self._last_C = None
        self._last_A = None
        self._last_G = None

        # Group and channel arguments
        assert 1 <= num_groups <= in_mod, \
            f"Number of groups must be between 1 and in_mod={in_mod}."
        self.num_groups = num_groups
        self.in_mod = in_mod
        self.use_mod = use_mod

        # Optional compatibilities scaling mechanism
        self.group_scaling = group_scaling

        # E_map embeds raw handcrafted mapping features
        E_map_cls = getattr(_local_modules, map_encoder)
        self.E_map = E_map_cls(in_map, nc_inner, **kwargs)

        # E_mod embeds the modality features in a space used as
        # values and to build attention scores in case use_mod=True
        self.E_mod = MLP([in_mod, in_mod, in_mod], bias=False)

        # E_mix combines the modality features from E_mod and
        # mapping features from E_map in case use_mod=True
        if self.use_mod:
            in_mix = nc_inner + in_mod
            out_mix = nc_inner
            mid_mix = nearest_power_of_2((in_mix + out_mix) / 2, out_mix * 2)
            self.E_mix = MLP([in_mix, mid_mix, out_mix], bias=False)

        # E_score computes the compatibility score for each feature
        # group, these are to be further normalized to produce
        # final attention scores
        self.E_score = nn.Linear(nc_inner, num_groups, bias=True)

        # Optional gating mechanism
        self.G = Gating(num_groups, bias=True) if gating else None

    def forward(self, x_main, x_mod, x_map, csr_idx):
        """
        :param x_main: N x F_main
        :param x_mod: V x F_mod
        :param x_map: V x F_map
        :param csr_idx:
        :return: x_pool, x_seen
        """
        # Compute mapping features : V x F_map
        x_map = self.E_map(x_map, csr_idx)

        # Compute values : V x F_mod
        x_mod = self.E_mod(x_mod)

        # Compute compatibilities (unscaled scores) : V x num_groups
        if self.use_mod:
            x_mix = self.E_mix(torch.cat([x_map, x_mod], dim=1))
            compatibilities = self.E_score(x_mix)
        else:
            compatibilities = self.E_score(x_map)

        # Compute attention scores : V x num_groups
        attentions = segment_softmax_csr(
            compatibilities, csr_idx, scaling=self.group_scaling)

        # Apply attention scores : P x F_mod
        x_pool = segment_csr(
            x_mod * expand_group_feat(attentions, self.num_groups, self.in_mod),
            csr_idx, reduce='sum')

        if self.G:
            # Compute pointwise gating for each group : P x num_groups
            gating = self.G(segment_csr(
                compatibilities, csr_idx, reduce='max'))

            # Apply gating to the features : P x F_mod
            x_pool = x_pool * expand_group_feat(
                gating, self.num_groups, self.in_mod)

        # Compute the boolean mask of seen points
        x_seen = csr_idx[1:] > csr_idx[:-1]

        # Optionally save outputs
        if self.save_last:
            self._last_x_map = x_map
            self._last_x_mod = x_mod
            self._last_idx = torch.arange(
                csr_idx.shape[0] - 1, device=x_map.device).repeat_interleave(
                csr_idx[1:] - csr_idx[:-1])
            self._last_view_num = csr_idx[1:] - csr_idx[:-1]
            self._last_C = expand_group_feat(
                compatibilities, self.num_groups, self.in_mod)
            self._last_A = expand_group_feat(
                attentions, self.num_groups, self.in_mod)
            if self.G:
                self._last_G = expand_group_feat(
                    gating, self.num_groups, self.in_mod)

        return x_pool, x_seen

    def extra_repr(self) -> str:
        repr_attr = ['num_groups', 'use_mod', 'group_scaling', 'save_last']
        return "\n".join([f'{a}={getattr(self, a)}' for a in repr_attr])


class QKVBimodalCSRPool(nn.Module, ABC):
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
    F_map = 3
    nc_qk = 2

    csr_idx = torch.LongTensor([0, 4, 4, 5, 10, 20])
    x_main = torch.rand(N, F_main)
    x_mod = torch.rand(V, F_mod)
    x = torch.rand(V, F_map)

    module = AttentiveBimodalCSRPool(in_main=F_main, in_map=F_map,
        in_mod=F_mod, nc_qk=nc_qk, use_map_min=False, use_map_max=False,
        use_map_num=False, gating=True, dim_scaling=True,
        group_scaling=True)

    module(x_main, x_mod, x, csr_idx)
    """

    def __init__(
            self, in_main=None, in_map=None, in_mod=None, num_groups=1,
            use_mod_q=False, use_mod_k=False, nc_qk=8, gating=True,
            dim_scaling=True, group_scaling=False, debug=False,
            save_last=False, nc_inner=32, map_encoder='DeepSetFeat', **kwargs):
        super(QKVBimodalCSRPool, self).__init__()

        # Default output feature size used for embeddings
        self.nc_inner = nc_inner

        # Optional mechanism to keep track of the outputs for debugging
        # or view-wise loss
        self.save_last = save_last
        self._last_x_map = None
        self._last_x_mod = None
        self._last_idx = None
        self._last_view_num = None
        self._last_Q = None
        self._last_K = None
        self._last_C = None
        self._last_A = None
        self._last_G = None
        self.debug = debug
        if debug:
            group_scaling = False
            dim_scaling = True
            nc_qk = 1
            in_map = 1
            in_mod = None

        # Group and channel arguments
        assert 1 <= num_groups <= in_mod, \
            f"Number of groups must be between 1 and in_mod={in_mod}."
        self.num_groups = num_groups
        self.in_mod = in_mod
        self.nc_qk = nc_qk
        self.use_mod_q = use_mod_q
        self.use_mod_k = use_mod_k

        # Optional compatibilities scaling mechanism
        self.dim_scaling = dim_scaling
        self.group_scaling = group_scaling

        # E_main embeds the main modality features in a space used as
        # queries and to build attention scores in case use_mod=True
        self.E_main = MLP([in_main, nc_inner, nc_inner], bias=False)

        # E_map embeds raw handcrafted mapping features
        E_map_cls = getattr(_local_modules, map_encoder)
        self.E_map = E_map_cls(in_map, nc_inner, **kwargs)

        # E_mod embeds the modality features in a space used as
        # values and to build attention scores in case use_mod=True
        self.E_mod = MLP([in_mod, in_mod, in_mod], bias=False)

        # E_mix_Q combines the modality features from E_mod and
        # mapping features from E_map in case use_mod_q=True
        if self.use_mod_q:
            in_mix = nc_inner + in_mod
            out_mix = nc_inner
            mid_mix = nearest_power_of_2((in_mix + out_mix) / 2, out_mix * 2)
            self.E_mix_Q = MLP([in_mix, mid_mix, out_mix], bias=False)

        # Queries computation module
        self.Q = nn.Linear(nc_inner, nc_qk * num_groups, bias=True)

        # E_mix_K combines the modality features from E_mod and
        # mapping features from E_map in case use_mod_q=True
        if self.use_mod_k:
            in_mix = nc_inner + in_mod
            out_mix = nc_inner
            mid_mix = nearest_power_of_2((in_mix + out_mix) / 2, out_mix * 2)
            self.E_mix_K = MLP([in_mix, mid_mix, out_mix], bias=False)

        # Keys computation module
        self.K = nn.Linear(nc_inner, nc_qk * num_groups, bias=True)

        # Optional gating mechanism
        self.G = Gating(num_groups, bias=True) if gating else None

    def forward(self, x_main, x_mod, x_map, csr_idx):
        """
        :param x_main: N x F_main
        :param x_mod: V x F_mod
        :param x_map: V x F_map
        :param csr_idx:
        :return: x_pool, x_seen
        """
        # Artificial x and x_mod
        if self.debug:
            device = x_map.device
            x_map = torch.rand((x_map.shape[0], 1), device=device)
            idx_destroyed = torch.where(x_map < 0.3)[0]
            x_mod[idx_destroyed] = torch.rand(
                (idx_destroyed.shape[0], *(x_mod.shape[1:])), device=device)

        # For MinkowskiEngine and TorchSparse SparseTensors
        if not isinstance(x_main, torch.Tensor):
            x_main = x_main.F

        # Compute main features : P x F_main
        x_main = self.E_main(x_main)

        # Compute mapping features : V x F_map
        x_map = self.E_map(x_map, csr_idx)

        # Compute modality features : V x F_mod
        x_mod = self.E_mod(x_mod)

        # Compute keys : V x (D x num_groups)
        if self.use_mod_k:
            x_mix = self.E_mix_K(torch.cat([x_map, x_mod], dim=1))
            keys = self.K(x_mix)
        else:
            keys = self.K(x_map)

        # Compute queries : V x (D x num_groups)
        if self.use_mod_q:
            # Expand x_main to views : V x F_main
            x_main_q = torch.repeat_interleave(
                x_main, csr_idx[1:] - csr_idx[:-1], dim=0)

            # Compute view-wise queries : V x (D x num_groups)
            x_mix = self.E_mix_Q(torch.cat([x_main_q, x_mod], dim=1))
            keys = self.Q(x_mix)
        else:
            # Compute pointwise queries : N x (D x num_groups)
            queries = self.Q(x_main)

            # Expand queries to views : V x (D x num_groups)
            queries = torch.repeat_interleave(
                queries, csr_idx[1:] - csr_idx[:-1], dim=0)

        # Compute compatibility scores : V x num_groups
        keys_per_group = keys.reshape(
            keys.shape[0], self.num_groups, self.nc_qk)
        queries_per_group = queries.reshape(
            queries.shape[0], self.num_groups, self.nc_qk)
        compatibilities = (keys_per_group * queries_per_group).sum(dim=2)

        # Optionally scale compatibilities by the number of key features
        if self.dim_scaling:
            compatibilities = compatibilities / math.sqrt(self.nc_qk)

        # Compute attention scores : V x num_groups
        attentions = segment_softmax_csr(
            compatibilities, csr_idx, scaling=self.group_scaling)

        # Apply attention scores : P x F_mod
        x_pool = segment_csr(
            x_mod * expand_group_feat(attentions, self.num_groups, self.in_mod),
            csr_idx, reduce='sum')

        if self.G:
            # Compute pointwise gating for each group : P x num_groups
            gating = self.G(segment_csr(
                compatibilities, csr_idx, reduce='max'))

            # Apply gating to the features : P x F_mod
            x_pool = x_pool * expand_group_feat(
                gating, self.num_groups, self.in_mod)

        # Compute the boolean mask of seen points
        x_seen = csr_idx[1:] > csr_idx[:-1]

        # Optionally save outputs
        if self.save_last:
            self._last_x_map = x_map
            self._last_x_mod = x_mod
            self._last_idx = torch.arange(
                csr_idx.shape[0] - 1, device=x_map.device).repeat_interleave(
                csr_idx[1:] - csr_idx[:-1])
            self._last_view_num = csr_idx[1:] - csr_idx[:-1]
            self._last_K = keys
            self._last_Q = queries
            self._last_C = compatibilities
            self._last_A = attentions
            if self.G:
                self._last_G = gating

        return x_pool, x_seen

    def extra_repr(self) -> str:
        repr_attr = ['dim_scaling', 'group_scaling', 'save_last']
        return "\n".join([f'{a}={getattr(self, a)}' for a in repr_attr])


class MinMaxDiffSetFeat(nn.Module, ABC):
    """Produce element-wise set features based on difference-to-min,
    difference-to-max or set size features.

    Inspired from:
        DeepSets: https://arxiv.org/abs/1703.06114
        PointNet: https://arxiv.org/abs/1612.00593
    """

    def __init__(
            self, d_in, d_out, use_min=True, use_max=True, use_num=False,
            **kwargs):
        super(MinMaxDiffSetFeat, self).__init__()

        # Initialize the MLPs
        self.d_in = d_in
        self.d_out = d_out
        self.use_min = use_min
        self.use_max = use_max
        self.use_num = use_num
        in_mlp = d_in * (1 + self.use_min + self.use_max) + self.use_num
        self.mlp = MLP([in_mlp, d_out, d_out], bias=False)

    def forward(self, x, csr_idx):
        # Optionally expand x with difference-to-min or
        # difference-to-max or group size features
        if self.use_min:
            x_map_min = x - segment_gather_csr(x, csr_idx, reduce='min')
        else:
            x_map_min = torch.empty(0, device=x.device)
        if self.use_max:
            x_map_max = x - segment_gather_csr(x, csr_idx, reduce='max')
        else:
            x_map_max = torch.empty(0, device=x.device)
        if self.use_num:
            # Heuristic to normalize in [0,1]
            x_map_num = torch.sqrt(1 / (csr_idx[1:] - csr_idx[:-1] + 1e-3))
            x_map_num = x_map_num.repeat_interleave(csr_idx[1:] - csr_idx[:-1])
            x_map_num = x_map_num.view(-1, 1)
        else:
            x_map_num = torch.empty(0, device=x.device)
        x_out = torch.cat([x, x_map_min, x_map_max, x_map_num], dim=1)
        x_out = self.mlp(x_out)
        return x_out

    def extra_repr(self) -> str:
        repr_attr = ['use_min', 'use_max', 'use_num']
        return "\n".join([f'{a}={getattr(self, a)}' for a in repr_attr])


class DeepSetFeat(nn.Module, ABC):
    """Produce element-wise set features based on shared learned
    features.

    Inspired from:
        DeepSets: https://arxiv.org/abs/1703.06114
        PointNet: https://arxiv.org/abs/1612.00593
    """

    _POOLING_MODES = ['max', 'mean', 'min', 'sum']
    _FUSION_MODES = ['residual', 'concatenation', 'both']

    def __init__(
            self, d_in, d_out, pool='max', fusion='concatenation',
            use_num=False, **kwargs):
        super(DeepSetFeat, self).__init__()

        # Initialize the set-pooling mechanism to aggregate features of
        # elements-level features to set-level features
        pool = pool.split('_')
        assert all([p in self._POOLING_MODES for p in pool]), \
            f"Unsupported pool='{pool}'. Expected elements of: " \
            f"{self._POOLING_MODES}"
        self.f_pool = lambda a, b: torch.cat([
            segment_csr(a, b, reduce=p) for p in pool], dim=-1)
        self.pool = pool

        # Initialize the fusion mechanism to merge set-level and
        # element-level features
        if fusion == 'residual':
            self.f_fusion = lambda a, b: a + b
        elif fusion == 'concatenation':
            self.f_fusion = lambda a, b: torch.cat((a, b), dim=-1)
        elif fusion == 'both':
            self.f_fusion = lambda a, b: torch.cat((a, a + b), dim=-1)
        else:
            raise NotImplementedError(
                f"Unknown fusion='{fusion}'. Please choose among "
                f"supported modes: {self._FUSION_MODES}.")
        self.fusion = fusion

        # Initialize the MLPs
        self.d_in = d_in
        self.d_out = d_out
        self.use_num = use_num
        self.mlp_elt_1 = MLP([d_in, d_out, d_out], bias=False)
        in_set_mlp = d_out * len(self.pool) + self.use_num
        self.mlp_set = MLP([in_set_mlp, d_out, d_out], bias=False)
        in_last_mlp = d_out if fusion == 'residual' else d_out * 2
        self.mlp_elt_2 = MLP([in_last_mlp, d_out, d_out], bias=False)

    def forward(self, x, csr_idx):
        x = self.mlp_elt_1(x)
        x_set = self.f_pool(x, csr_idx)
        if self.use_num:
            # Heuristic to normalize in [0,1]
            set_num = torch.sqrt(1 / (csr_idx[1:] - csr_idx[:-1] + 1e-3))
            x_set = torch.cat((x_set, set_num.view(-1, 1)), dim=1)
        x_set = self.mlp_set(x_set)
        x_set = gather_csr(x_set, csr_idx)
        x_out = self.f_fusion(x, x_set)
        x_out = self.mlp_elt_2(x_out)
        return x_out

    def extra_repr(self) -> str:
        repr_attr = ['pool', 'fusion', 'use_num']
        return "\n".join([f'{a}={getattr(self, a)}' for a in repr_attr])


class Gating(nn.Module):
    """Rectified-tanh gating mechanism with learnable linear correction."""
    def __init__(self, num_groups, weight=True, bias=True):
        super(Gating, self).__init__()
        self.num_groups = num_groups
        self.weight = nn.Parameter(torch.ones(1, num_groups)) if weight \
            else None
        self.bias = nn.Parameter(torch.zeros(1, num_groups)) if bias else None

    def forward(self, x):
        if self.weight is not None:
            x = self.weight * x
        if self.bias is not None:
            x = x + self.bias
        return torch.tanh(torch.relu(x)).view(-1, self.num_groups).squeeze(1)

    def extra_repr(self) -> str:
        return f'num_groups={self.num_groups}, ' \
            f'weight={self.weight is not None}, bias={self.bias is not None}'


def nearest_power_of_2(x, min_power=16):
    """Local helper to find the nearest power of 2 of a given number.
    The `min_power` parameter puts a minimum threshold for the returned
    power.
    """
    x = int(x)

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
    sizes = torch.full(
        (num_groups,), math.floor(num_elements / num_groups),
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
def segment_softmax_csr(src: torch.Tensor, csr_idx: torch.Tensor,
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
    dense_idx = torch.arange(n_groups).to(src.device).repeat_interleave(
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
        num_per_src_element = torch.repeat_interleave(
            sqrt_num_per_index, num_per_index)
        if src.dim() > 1:
            num_per_src_element = num_per_src_element.view(-1, 1).repeat(
                1, src.shape[1])

        centered_scores = centered_scores / num_per_src_element

    # Compute the numerators
    centered_scores_exp = centered_scores.exp()

    # Compute the denominators
    sum_per_index = segment_csr(centered_scores_exp, csr_idx, reduce='sum')
    normalizing_constants = sum_per_index.add_(eps).gather(0, dense_idx)

    return centered_scores_exp.div(normalizing_constants)


@torch.jit.script
def gather_csr(src: torch.Tensor, csr_idx: torch.Tensor) -> torch.Tensor:
    """Gather index-level src values into element-level values based on
    CSR indices.

    When applied to the output or segment_csr, this redistributes the
    reduced values to the appropriate segment_csr input elements.
    """
    if not torch.is_floating_point(src):
        raise ValueError(
            '`gather_csr` can only be computed over tensors with '
            'floating point data types.')
    if csr_idx.dim() != 1:
        raise ValueError(
            '`gather_csr` can only be computed over 1D CSR indices.')
    if src.dim() > 2:
        raise NotImplementedError(
            '`gather_csr` can only be computed over 1D or 2D source '
            'tensors.')

    # Compute dense indices from CSR indices
    n_groups = csr_idx.shape[0] - 1
    dense_idx = torch.arange(n_groups).to(src.device).repeat_interleave(
        csr_idx[1:] - csr_idx[:-1])
    if src.dim() > 1:
        dense_idx = dense_idx.view(-1, 1).repeat(1, src.shape[1])

    # Center scores maxima near 1 for computation precision
    return src.gather(0, dense_idx)


@torch.jit.script
def segment_gather_csr(src: torch.Tensor, csr_idx: torch.Tensor,
                       reduce: str = 'sum') -> torch.Tensor:
    """Compute the reduced value between same-index elements, for CSR
    indices, and redistribute them to input elements.
    """
    # Reduce with segment_csr
    reduced_per_index = segment_csr(src, csr_idx, reduce=reduce)

    # Expand with gather_csr
    reduced_per_src_element = gather_csr(reduced_per_index, csr_idx)

    return reduced_per_src_element


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
