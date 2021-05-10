from abc import ABC

import torch
import torch.nn as nn
from torch_points3d.core.multimodal.data import MODALITY_NAMES
from torch_points3d.core.common_modules.base_modules import Identity
from torchsparse.nn.functional import sphash, sphashquery
import torch_scatter

try:
    import MinkowskiEngine as me
except:
    me = None
try:
    import torchsparse as ts
except:
    ts = None


class MultimodalBlockDown(nn.Module, ABC):
    """Multimodal block with downsampling that looks like:

                 -- 3D Conv ---- Merge i -- 3D Conv --
    MMData IN          ...        |                       MMData OUT
                 -- Mod i Conv --|--------------------
                       ...
    """

    def __init__(self, down_block, conv_block, **kwargs):
        """Build the Multimodal module from already-instantiated
        modules. Modality-specific modules are expected to be passed in
        dictionaries holding fully-fledged UnimodalBranch modules.
        """
        # BaseModule initialization
        super(MultimodalBlockDown, self).__init__()

        # Blocks for the implicitly main modality: 3D
        self.down_block = down_block if down_block is not None else Identity()
        self.conv_block = conv_block if conv_block is not None else Identity()

        # Initialize the dict holding the conv and merge blocks for all
        # modalities
        self._modalities = []
        self._init_from_kwargs(**kwargs)

        # Expose the 3D down_conv .sampler attribute (for
        # UnwrappedUnetBasedModel)
        # TODO this is for KPConv, is it doing the intended, is it
        #  needed at all ?
        self.sampler = [getattr(self.down_block, "sampler", None),
                        getattr(self.conv_block, "sampler", None)]

    def _init_from_kwargs(self, **kwargs):
        """Kwargs are expected to carry fully-fledged modality-specific
        UnimodalBranch modules.
        """
        for m in kwargs.keys():
            assert (m in MODALITY_NAMES), \
                f"Invalid kwarg modality '{m}', expected one of " \
                f"{MODALITY_NAMES}."
            assert isinstance(kwargs[m], (UnimodalBranch, nn.Identity)), \
                f"Expected a UnimodalBranch module for '{m}' modality " \
                f"but got {type(kwargs[m])} instead."
            setattr(self, m, kwargs[m])
            self._modalities.append(m)

    @property
    def modalities(self):
        return self._modalities

    @property
    def num_modalities(self):
        return len(self.modalities) + 1

    def forward(self, mm_data_dict):
        """
        Forward pass of the MultiModalBlockDown.

        Expects a tuple of 3D data (Data, SparseTensor, etc.) destined
        for the 3D convolutional modules, and a dictionary of
        modality-specific data equipped with corresponding mappings.
        """
        # Conv on the main 3D modality - assumed to reduce 3D resolution
        mm_data_dict = self.forward_3d_block_down(mm_data_dict, self.down_block)

        for m in self.modalities:
            # TODO: does the modality-driven sequence of updates on x_3d
            #  and x_seen affect the modality behavior ? Should the shared
            #  3D information only be updated once all modality branches
            #  have been run on the same input ?
            mod_branch = getattr(self, m)
            mm_data_dict = mod_branch(mm_data_dict, m)

        # Conv on the main 3D modality
        mm_data_dict = self.forward_3d_block_down(mm_data_dict, self.conv_block)

        return mm_data_dict

    @staticmethod
    def forward_3d_block_down(mm_data_dict, block):
        """
        Wrapper method to apply the forward pass on a 3D down conv
        block while preserving modality-specific mappings.

        This both runs the forward method of the input block but also
        catches the reindexing scheme, in case a sampling or sparse
        strided convolution is applied in the 3D conv block.

        For MinkowskiEngine or TorchSparse sparse tensors, the
        reindexing is recovered from the input/output coordinates. If
        no strided convolution was applied, the indexing stays the same
        and a None index is returned. Otherwise, the returned index
        maps indices as follows: i -> idx[i].

        For non-sparse convolutions, the reindexing is carried by the
        sampler's 'last_index' attribute. If no sampling was applied,
        the indexing stays the same and a None index is returned.
        Otherwise, the returned index carries the indices of the
        selected points with respect to their input order.
        """
        # Leave the input untouched if the 3D conv block is Identity
        if isinstance(block, nn.Identity):
            return mm_data_dict

        # Unpack the multimodal data dictionary
        x_3d = mm_data_dict['x_3d']
        x_seen = mm_data_dict['x_seen']

        # Initialize index and indexation mode
        idx = None
        mode = 'pick'

        # Non-sparse forward and reindexing
        if isinstance(x_3d, torch.Tensor):
            # Forward pass on the block while keeping track of the
            # sampler indices
            block.sampler.last_idx = None
            idx_ref = torch.arange(x_3d.shape[0])
            x_3d = block(x_3d)
            idx_sample = block.sampler.last_idx
            if (idx_sample == idx_ref).all():
                idx = None
            else:
                idx = idx_sample
            mode = 'pick'

        # MinkowskiEngine forward and reindexing
        elif me is not None and isinstance(x_3d, me.SparseTensor):
            mode = 'merge'

            # Forward pass on the block while keeping track of the
            # stride levels
            stride_in = x_3d.tensor_stride[0]
            x_3d = block(x_3d)
            stride_out = x_3d.tensor_stride[0]

            if stride_in == stride_out:
                idx = None
            else:
                src, target = x_3d.coords_man.get_coords_map(
                    stride_in, stride_out)
                idx = target[src.argsort()]

        # TorchSparse forward and reindexing
        elif ts is not None and isinstance(x_3d, ts.SparseTensor):
            # Forward pass on the block while keeping track of the
            # stride levels
            stride_in = x_3d.s
            x_3d = block(x_3d)
            stride_out = x_3d.s

            if stride_in == stride_out:
                idx = None
            else:
                # To compute the reindexing of the sparse voxels with
                # torchsparse, we need to make use of the torchsparse
                # sphashquery function to compare sets of coordinates at
                # the same resolution. However, when changing resolution
                # we must be careful to voxelize spatial points but
                # leave the batch indices untouched. For torchsparse,
                # the batch indices are stored in the last column of
                # the coords tensor (unlike MinkowskiEngine which
                # stores batch indices in the first column). Hence we
                # assume here that coordinates to have shape (N x 4) and
                # batch indices to lie in the last column.
                assert x_3d.C.shape[1] == 4, \
                    f"Sparse coordinates are expected to have shape " \
                    f"(N x 4), with batch indices in the first column and " \
                    f"3D spatial coordinates in the following ones. Yet, " \
                    f"received coordinates tensor with shape {x_3d.C.shape} " \
                    f"instead."
                in_coords = x_3d.coord_maps[stride_in]
                in_coords[:, :3] = ((in_coords[:, :3].float() / stride_out
                                     ).floor() * stride_out).int()
                out_coords = x_3d.coord_maps[stride_out]
                idx = sphashquery(sphash(in_coords), sphash(out_coords))
            mode = 'merge'

        else:
            raise NotImplementedError(
                f"Unsupported format for x_3d: {type(x_3d)}. If you are trying "
                f"to use MinkowskiEngine or TorchSparse, make sure those are "
                f"properly installed.")

        # Update seen 3D points indices
        if x_seen is not None and idx is not None:
            if mode == 'pick':
                x_seen = x_seen[idx]
            else:
                x_seen = torch_scatter.scatter(x_seen, idx, reduce='sum')

        # Update the multimodal data dictionary
        mm_data_dict['x_3d'] = x_3d
        mm_data_dict['x_seen'] = x_seen

        # Update modality data and mappings wrt new point indexing
        for m in mm_data_dict['modalities'].keys():
            mm_data_dict['modalities'][m] = \
                mm_data_dict['modalities'][m].select_points(idx, mode=mode)

        return mm_data_dict


class UnimodalBranch(nn.Module, ABC):
    """Unimodal block with downsampling that looks like:

    IN 3D   ------------------------------------           --  OUT 3D
                                   \            \         /
                       Atomic Pool -- View Pool -- Fusion
                     /
    IN Mod  -- Conv -----------------------------------------  OUT Mod

    The convolution may be a down-convolution or preserve input shape.
    However, up-convolutions are not supported, because reliable the
    mappings cannot be inferred when increasing resolution.
    """

    def __init__(self, conv, atomic_pool, view_pool, fusion, drop_3d=0,
                 drop_mod=0, keep_last_view=False):
        super(UnimodalBranch, self).__init__()
        self.conv = conv
        self.atomic_pool = atomic_pool
        self.view_pool = view_pool
        self.fusion = fusion
        self.drop_3d = nn.Dropout(p=drop_3d, inplace=True) \
            if drop_3d is not None and drop_3d > 0 \
            else None
        self.drop_mod = nn.Dropout(p=drop_mod, inplace=True) \
            if drop_mod is not None and drop_mod > 0 \
            else None
        self.keep_last_view = keep_last_view

    def forward(self, mm_data_dict, modality):
        # Unpack the multimodal data dictionary
        x_3d = mm_data_dict['x_3d']
        mod_data = mm_data_dict['modalities'][modality]

        # Check whether the modality carries multi-setting data
        has_multi_setting = isinstance(mod_data.x, list)

        # Conv on the modality data. The modality data holder
        # carries a feature tensor per modality settings. Hence the
        # modality features are provided as a list of tensors.
        # Update modality features and mappings wrt modality scale.
        # Note that convolved features are preserved in the modality
        # data holder, to be later used in potential downstream
        # modules.
        if self.conv:
            if has_multi_setting:
                for i in range(len(mod_data)):
                    mod_data[i].update_features_and_scale(
                        self.conv(mod_data[i].x, reset_dropout=(i == 0)))
            else:
                mod_data = mod_data.update_features_and_scale(
                    self.conv(mod_data.x, reset_dropout=True))

        # Extract CSR-arranged atomic features from the feature maps
        # of each input modality setting
        if has_multi_setting:
            x_mod = [x[idx]
                     for x, idx
                     in zip(mod_data.x, mod_data.feature_map_indexing)]
        else:
            x_mod = mod_data.x[mod_data.feature_map_indexing]

        # Atomic pooling of the modality features on each
        # separate setting
        if has_multi_setting:
            x_mod = [self.atomic_pool(x_3d, x, None, a_idx)[0]
                     for x, a_idx
                     in zip(x_mod, mod_data.atomic_csr_indexing)]
        else:
            x_mod = self.atomic_pool(x_3d, x_mod, None,
                                     mod_data.atomic_csr_indexing)[0]

        # For multi-setting data, concatenate view-level features from
        # each input modality setting and sort them to a CSR-friendly
        # order wrt 3D points features
        if has_multi_setting:
            idx_sorting = mod_data.view_cat_sorting
            x_mod = torch.cat(x_mod, dim=0)[idx_sorting]
            x_proj = torch.cat(mod_data.mapping_features, dim=0)[idx_sorting]

        # View pooling of the atomic-pooled modality features

        if has_multi_setting:
            csr_idx = mod_data.view_cat_csr_indexing
        else:
            csr_idx = mod_data.view_csr_indexing
        if self.keep_last_view:
            # Here we keep track of the latest x_mod, x_proj and csr_idx
            # in the modality data so as to recover it at the end of a
            # multimodal encoder or UNet. This is necessary when
            # training on a view-level loss.
            mod_data.last_view_x_mod = x_mod
            mod_data.last_view_x_proj = x_proj
            mod_data.last_view_csr_idx = csr_idx
        x_mod, x_seen = self.view_pool(x_3d, x_mod, x_proj, csr_idx)

        # Dropout 3D or modality features
        if self.drop_3d:
            if isinstance(x_3d, torch.Tensor):
                x_3d = self.drop_3d(x_3d)
            elif x_3d is not None:
                x_3d.F = self.drop_3d(x_3d.F)
        if self.drop_mod:
            x_mod = self.drop_mod(x_mod)
            if self.keep_last_view:
                mod_data.last_view_x_mod = self.drop_mod(mod_data.last_view_x_mod)

        # Fuse the modality features into the 3D points features
        x_3d = self.fusion(x_3d, x_mod)

        # Update the multimodal data dictionary
        # TODO: does the modality-driven sequence of updates on x_3d
        #  and x_seen affect the modality behavior ? Should the shared
        #  3D information only be updated once all modality branches
        #  have been run on the same input ?
        mm_data_dict['x_3d'] = x_3d
        mm_data_dict['modalities'][modality] = mod_data
        if mm_data_dict['x_seen'] is None:
            mm_data_dict['x_seen'] = x_seen
        else:
            mm_data_dict['x_seen'] = torch.logical_or(
                x_seen, mm_data_dict['x_seen'])

        return mm_data_dict
