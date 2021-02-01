import torch
import torch.nn as nn
from torch_points3d.datasets.multimodal.data import MODALITY_NAMES
from torch_points3d.core.common_modules.base_modules import Identity
import MinkowskiEngine as me
import torchsparse as ts
from torchsparse.nn.functional import sphash, sphashquery


class MultimodalBlockDown(nn.Module):
    """Multimodal block with downsampling that looks like:

    IN MMData    -- 3D Down Conv -- Merge 1 -- Merge i -- 3D Conv --    OUT MMData
                                      |          |
                 -- Mod 1 Down Conv --          |
                          ...                  |
                 -- Mod i Down Conv -----------
                          ...
    """
    def __init__(self, down_block, conv_block, **kwargs):
        """Build the Multimodal module from already-instantiated modules.
        Modality-specific modules are expected to be passed in dictionaries
        holding the conv and merge modules under 'conv' and 'merge' keys.
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

        # TODO : check layers compatibility

    def _init_from_kwargs(self, **kwargs):
        """Kwargs are expected to carry fully-fledged modality-specific
        UnimodalBranch modules.
        """
        for m in kwargs.keys():
            assert (m in MODALITY_NAMES), \
                f"Invalid kwarg modality '{m}', expected one of " \
                f"{MODALITY_NAMES}."
            assert isinstance(kwargs[m], UnimodalBranch), \
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

    def forward(self, mm_data_tuple):
        """
        Forward pass of the MultiModalBlockDown.

        Expects a tuple of 3D data (Data, SparseTensor, etc.) destined
        for the 3D convolutional modules, and a dictionary of
        modality-specific data equipped with corresponding mappings.
        """
        # Unpack the multimodal data tuple
        x_3d, mod_dict = mm_data_tuple

        # Conv on the main 3D modality - assumed to reduce 3D resolution
        x_3d, mod_dict = self.forward_3d_block_down(
            x_3d, mod_dict, self.down_block)

        for m in self.modalities:
            mod_branch = getattr(self, m)
            x_3d, mod_dict[m] = mod_branch(x_3d, mod_dict[m])

        # Conv on the main 3D modality
        x_3d, mod_dict = self.forward_3d_block_down(
            x_3d, mod_dict, self.conv_block)

        return tuple((x_3d, mod_dict))

    @staticmethod
    def forward_3d_block_down(x_3d, mod_dict, block):
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
            return x_3d, mod_dict

        # Initialize index and indexation mode
        idx = None
        mode = 'pick'

        # MinkowskiEngine forward and reindexing
        if isinstance(x_3d, me.SparseTensor):
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
        elif isinstance(x_3d, ts.SparseTensor):
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

        # Non-sparse forward and reindexing
        else:
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

        # Update modality data and mappings wrt new point indexing
        for m in mod_dict.keys():
            mod_dict[m] = mod_dict[m].select_points(idx, mode=mode)

        return x_3d, mod_dict


class UnimodalBranch(nn.Module):
    """Unimodal block with downsampling that looks like:

    IN 3D    ------------------------------------
                                    \            \
    IN Mod   -- Conv -- Atomic Pool -- View Pool -- Fusion --    OUT 3D
                      \
                       --------------------------------------    OUT Mod

    The convolution may be a down-convolution or preserve input shape.
    However, up-convolutions are not supported, because reliable the
    mappings cannot be inferred when increasing resolution.
    """

    def __init__(self, conv, atomic_pool, view_pool, fusion):
        super(UnimodalBranch, self).__init__()
        self.conv = conv if conv is not None else Identity()
        self.atomic_pool = atomic_pool
        self.view_pool = view_pool
        self.fusion = fusion

    def forward(self, x_3d, mod_data):
        # Check whether the modality carries multi-setting data
        is_multi = isinstance(mod_data.x, list)

        # Conv on the modality data. The modality data holder
        # carries a feature tensor per modality settings. Hence the
        # modality features are provided as a list of tensors.
        if is_multi:
            x_mod = [self.conv(x) for x in mod_data.x]
        else:
            x_mod = self.conv(mod_data.x)

        # Update modality features and mappings wrt modality scale.
        # Note that convolved features are preserved in the modality
        # data holder, to be later used in potential downstream
        # modules.
        mod_data = mod_data.update_features_and_scale(x_mod)

        # Extract CSR-arranged atomic features from the feature maps
        # of each input modality setting
        if is_multi:
            x_mod = [x[idx]
                     for x, idx
                     in zip(x_mod, mod_data.feature_map_indexing)]
        else:
            x_mod = x_mod[mod_data.feature_map_indexing]

        # Atomic pooling of the modality features on each
        # separate setting
        if is_multi:
            x_mod = [self.atomic_pool(x_3d, x, a_idx)
                     for x, a_idx
                     in zip(x_mod, mod_data.atomic_csr_indexing)]
        else:
            x_mod = self.atomic_pool(x_3d, x_mod, mod_data.atomic_csr_indexing)

        # For multi-setting data, concatenate view-level features from
        # each input modality setting and sort them to a CSR-friendly
        # order wrt 3D points features
        if is_multi:
            x_mod = torch.cat(x_mod, dim=0)
            x_mod = x_mod[mod_data.view_cat_sorting]

        # View pooling of the joint modality features
        if is_multi:
            x_mod = self.view_pool(x_3d, x_mod, mod_data.view_cat_csr_indexing)
        else:
            x_mod = self.view_pool(x_3d, x_mod, mod_data.view_csr_indexing)

        # Fuse the modality features into the 3D points features
        x_3d = self.fusion(x_3d, x_mod)

        return x_3d, mod_data
