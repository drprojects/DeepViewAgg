from abc import ABC

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from torch_points3d.core.multimodal.data import MODALITY_NAMES
from torch_points3d.core.common_modules.base_modules import Identity, \
    BaseModule
from torch_points3d.modules.multimodal.dropout import ModalityDropout
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

    def __init__(self, block_1, block_2, **kwargs):
        """Build the Multimodal module from already-instantiated
        modules. Modality-specific modules are expected to be passed in
        dictionaries holding fully-fledged UnimodalBranch modules.
        """
        # BaseModule initialization
        super(MultimodalBlockDown, self).__init__()

        # Blocks for the implicitly main modality: 3D
        self.block_1 = block_1 if block_1 is not None else Identity()
        self.block_2 = block_2 if block_2 is not None else Identity()

        # Initialize the dict holding the conv and merge blocks for all
        # modalities
        self._modalities = []
        self._init_from_kwargs(**kwargs)

        # Expose the 3D convs .sampler attribute (for
        # UnwrappedUnetBasedModel)
        # TODO this is for KPConv, is it doing the intended, is it
        #  needed at all ?
        self.sampler = [
            getattr(self.block_1, "sampler", None),
            getattr(self.block_2, "sampler", None)]

    def _init_from_kwargs(self, **kwargs):
        """Kwargs are expected to carry fully-fledged modality-specific
        UnimodalBranch modules.
        """
        for m in kwargs.keys():
            assert (m in MODALITY_NAMES), \
                f"Invalid kwarg modality '{m}', expected one of " \
                f"{MODALITY_NAMES}."
            assert isinstance(kwargs[m], (UnimodalBranch, IdentityBranch)), \
                f"Expected a UnimodalBranch module for '{m}' modality " \
                f"but got {type(kwargs[m])} instead."
            setattr(self, m, kwargs[m])
            self._modalities.append(m)

    @property
    def modalities(self):
        return self._modalities

    def forward(self, mm_data_dict):
        """
        Forward pass of the MultiModalBlockDown.

        Expects a tuple of 3D data (Data, SparseTensor, etc.) destined
        for the 3D convolutional modules, and a dictionary of
        modality-specific data equipped with corresponding mappings.
        """
        # Conv on the main 3D modality - assumed to reduce 3D resolution
        mm_data_dict = self.forward_3d_block_down(
            mm_data_dict, self.block_1)

        for m in self.modalities:
            # TODO: does the modality-driven sequence of updates on x_3d
            #  and x_seen affect the modality behavior ? Should the shared
            #  3D information only be updated once all modality branches
            #  have been run on the same input ?
            mod_branch = getattr(self, m)
            mm_data_dict = mod_branch(mm_data_dict, m)

        # Conv on the main 3D modality
        mm_data_dict = self.forward_3d_block_down(
            mm_data_dict, self.block_2)

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
        if isinstance(block, Identity):
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


class MultimodalBlockUp(nn.Module, ABC):
    """Multimodal block with downsampling that looks like:

                 -- 3D Conv ---- Merge i -- 3D Conv --
    MMData IN          ...        |                       MMData OUT
                 -- Mod i Conv --|--------------------
                       ...
    """


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

    def __init__(
            self, conv, atomic_pool, view_pool, fusion, drop_3d=0, drop_mod=0,
            hard_drop=False, keep_last_view=False, checkpointing='',
            out_channels=None, interpolate=False):
        super(UnimodalBranch, self).__init__()
        self.conv = conv
        self.atomic_pool = atomic_pool
        self.view_pool = view_pool
        self.fusion = fusion
        drop_cls = ModalityDropout if hard_drop else nn.Dropout
        self.drop_3d = drop_cls(p=drop_3d, inplace=False) \
            if drop_3d is not None and drop_3d > 0 \
            else None
        self.drop_mod = drop_cls(p=drop_mod, inplace=True) \
            if drop_mod is not None and drop_mod > 0 \
            else None
        self.keep_last_view = keep_last_view
        self._out_channels = out_channels
        self.interpolate = interpolate

        # Optional checkpointing to alleviate memory at train time.
        # Character rules:
        #     c: convolution
        #     a: atomic pooling
        #     v: view pooling
        #     f: fusion
        assert not checkpointing or isinstance(checkpointing, str),\
            f'Expected checkpointing to be of type str but received ' \
            f'{type(checkpointing)} instead.'
        self.checkpointing = ''.join(set('cavf').intersection(set(checkpointing)))

    @property
    def out_channels(self):
        if self._out_channels is None:
            raise ValueError(
                f'{self.__class__.__name__}.out_channels has not been '
                f'set. Please set it to allow inference even when the '
                f'modality has no data.')
        return self._out_channels

    def forward(self, mm_data_dict, modality):
        # Unpack the multimodal data dictionary. Specific treatment for
        # MinkowskiEngine and TorchSparse SparseTensors
        is_sparse_3d = not isinstance(
            mm_data_dict['x_3d'], (torch.Tensor, type(None)))
        x_3d = mm_data_dict['x_3d'].F if is_sparse_3d else mm_data_dict['x_3d']
        mod_data = mm_data_dict['modalities'][modality]

        # Check whether the modality carries multi-setting data
        is_multi_shape = isinstance(mod_data.x, list)

        # If the modality has no data mapped to the current 3D points,
        # ignore the branch forward. `self.out_channels` will guide us
        # on how to replace expected modality features
        if is_multi_shape and all([e.x.shape[0] == 0 for e in mod_data]) \
                or is_multi_shape and len(mod_data) == 0 \
                or not is_multi_shape and mod_data.x.shape[0] == 0:

            # Prepare the channel sizes
            nc_out = self.out_channels
            nc_3d = x_3d.shape[1]
            nc_2d = nc_out - nc_3d if nc_out > nc_3d else nc_3d

            # Make sure we have a valid `self.out_channels` so we can
            # simulate the forward without any modality data
            if nc_out < nc_3d:
                raise ValueError(
                    f'{self.__class__.__name__}.out_channels is smaller than '
                    f'number of features in x_3d: {nc_out} < {nc_3d}')

            # No points are seen
            # x_seen = torch.zeros(nc_3d, dtype=torch.bool)

            # Modify the feature dimension of mod_data to simulate
            # convolutions too
            if not is_multi_shape:
                mod_data.x = mod_data.x[:, [0]].repeat_interleave(nc_2d, dim=1)
            elif len(mod_data) > 0:
                mod_data.x = [
                    x[:, [0]].repeat_interleave(nc_2d, dim=1)
                    for x in mod_data.x]

            # For concatenation fusion, create zero features to
            # 'simulate' concatenation of modality features to x_3d
            if nc_out > nc_3d:
                zeros = torch.zeros_like(x_3d[:, [0]])
                zeros = zeros.repeat_interleave(nc_2d, dim=1)
                x_3d = torch.cat((x_3d, zeros), dim=1)

            # Return the modified multimodal data dictionary despite the
            # absence of modality features
            if is_sparse_3d:
                mm_data_dict['x_3d'].F = x_3d
            else:
                mm_data_dict['x_3d'] = x_3d
            mm_data_dict['modalities'][modality] = mod_data
            # if mm_data_dict['x_seen'] is None:
            #     mm_data_dict['x_seen'] = x_seen
            # else:
            #     mm_data_dict['x_seen'] = torch.logical_or(
            #         x_seen, mm_data_dict['x_seen'])

            return mm_data_dict

        # If the modality has a data list format and that one of the
        # items is an empty feature map, run a recursive forward on the
        # mm_data_dict with these problematic items discarded. This is
        # necessary whenever an element of the batch has no mappings to
        # the modality
        if is_multi_shape and any([e.x.shape[0] == 0 for e in mod_data]):

            # Remove problematic elements from mod_data
            num = len(mod_data)
            removed = {
                i: e for i, e in enumerate(mod_data) if e.x.shape[0] == 0}
            indices = [i for i in range(num) if i not in removed.keys()]
            mm_data_dict['modalities'][modality] = mod_data[indices]

            # Run forward recursively
            mm_data_dict = self.forward(mm_data_dict, modality)

            # Restore problematic elements. This is necessary if we need
            # to restore the initial batch elements with methods such as
            # `MMBatch.to_mm_data_list`
            mod_data = mm_data_dict['modalities'][modality]
            kept = {k: e for k, e in zip(indices, mod_data)}
            joined = {**kept, **removed}
            mod_data = mod_data.__class__([joined[i] for i in range(num)])
            mm_data_dict['modalities'][modality] = mod_data

            return mm_data_dict

        # Conv on the modality data. The modality data holder
        # carries a feature tensor per modality settings. Hence the
        # modality features are provided as a list of tensors.
        # Update modality features and mappings wrt modality scale. If
        # `self.interpolate`, do not modify the mappings' scale, so that
        # the features can be interpolated to the input resolution.
        # Note that convolved features are preserved in the modality
        # data holder, to be later used in potential downstream
        # modules.
        if self.conv:
            if is_multi_shape:
                for i in range(len(mod_data)):
                    if 'c' in self.checkpointing:
                        # Need to set requires_grad for input tensor
                        # because checkpointing the first layer breaks
                        # the gradients
                        mod_data[i].x.requires_grad_()
                        reset = torch.BoolTensor([i == 0])
                        mod_x = checkpoint(self.conv, mod_data[i].x, reset)
                    else:
                        mod_x = self.conv(mod_data[i].x, i == 0)
                    mod_data[i].update_x(mod_x)
            else:
                if 'c' in self.checkpointing:
                    # Need to set requires_grad for input tensor
                    # because checkpointing the first layer breaks
                    # the gradients
                    mod_data.x.requires_grad_()
                    reset = torch.BoolTensor([True])
                    mod_x = checkpoint(self.conv, mod_data.x, reset)
                else:
                    mod_x = self.conv(mod_data.x, True)
                mod_data.update_x(mod_x)

        # Extract CSR-arranged atomic features from the feature maps
        # of each input modality setting
        if is_multi_shape:
            if not self.interpolate:
                x_mod = [
                    x[idx]
                    for x, idx
                    in zip(mod_data.x, mod_data.feature_map_indexing)]
            else:
                raise NotImplementedError
        else:
            if not self.interpolate:
                # Recover the downscaled mapping for indexing the
                # modality feature map in its lower resolution, in case
                # the resolution was reduced
                # TODO: support upsampling
                # TODO: support anisotropic sampling
                scale_x = mod_data.crop_size[0] / mod_data.x.shape[3]
                scale_y = mod_data.crop_size[1] / mod_data.x.shape[2]
                scale = max(scale_x, scale_y)
                mapping = mod_data.mappings.downscale_images(scale)

                # TODO: we want to hide this away and let images deal with it themselves
                x_mod = mod_data.x[mod_data.feature_map_indexing]
            else:
                # Sparse interpolation
                grid_sparse = grid[:, np.random.choice(np.arange(res), 5),
                              np.random.choice(np.arange(res), 5)].unsqueeze(2)
                out_sparse = torch.nn.functional.grid_sample(image.unsqueeze(0), grid_sparse,
                                                             align_corners=False).squeeze(0).squeeze(-1)

        # Atomic pooling of the modality features on each
        # separate setting
        if is_multi_shape:
            if 'a' in self.checkpointing:
                x_mod = [
                    checkpoint(self.atomic_pool, x_3d, x, None, a_idx)
                    for x, a_idx in zip(x_mod, mod_data.atomic_csr_indexing)]
            else:
                x_mod = [
                    self.atomic_pool(x_3d, x, None, a_idx)
                    for x, a_idx in zip(x_mod, mod_data.atomic_csr_indexing)]
        elif 'a' in self.checkpointing:
            x_mod = checkpoint(
                self.atomic_pool, x_3d, x_mod, None,
                mod_data.atomic_csr_indexing)
        else:
            x_mod = self.atomic_pool(
                x_3d, x_mod, None, mod_data.atomic_csr_indexing)

        # For multi-setting data, concatenate view-level features from
        # each input modality setting and sort them to a CSR-friendly
        # order wrt 3D points features
        if is_multi_shape:
            idx_sorting = mod_data.view_cat_sorting
            x_mod = torch.cat(x_mod, dim=0)[idx_sorting]
            x_map = torch.cat(mod_data.mapping_features, dim=0)[idx_sorting]

        # View pooling of the atomic-pooled modality features
        if is_multi_shape:
            csr_idx = mod_data.view_cat_csr_indexing
        else:
            csr_idx = mod_data.view_csr_indexing
        if self.keep_last_view:
            # Here we keep track of the latest x_mod, x_map and csr_idx
            # in the modality data so as to recover it at the end of a
            # multimodal encoder or UNet. This is necessary when
            # training on a view-level loss.
            mod_data.last_view_x_mod = x_mod
            mod_data.last_view_x_map = x_map
            mod_data.last_view_csr_idx = csr_idx
        if 'v' in self.checkpointing:
            x_mod = checkpoint(self.view_pool, x_3d, x_mod, x_map, csr_idx)
        else:
            x_mod = self.view_pool(x_3d, x_mod, x_map, csr_idx)

        # Compute the boolean mask of seen points
        x_seen = csr_idx[1:] > csr_idx[:-1]

        # Dropout 3D or modality features
        if self.drop_3d:
            x_3d = self.drop_3d(x_3d)
        if self.drop_mod:
            x_mod = self.drop_mod(x_mod)
            if self.keep_last_view:
                mod_data.last_view_x_mod = self.drop_mod(mod_data.last_view_x_mod)

        # Fuse the modality features into the 3D points features
        if 'f' in self.checkpointing:
            x_3d = checkpoint(self.fusion, x_3d, x_mod)
        else:
            x_3d = self.fusion(x_3d, x_mod)

        # In case it has not been provided at initialization, save the
        # output channel size. This is useful for when a batch has no
        # modality data
        self._out_channels = x_3d.shape[1]

        # Update the multimodal data dictionary
        # TODO: does the modality-driven sequence of updates on x_3d
        #  and x_seen affect the modality behavior ? Should the shared
        #  3D information only be updated once all modality branches
        #  have been run on the same input ?
        if is_sparse_3d:
            mm_data_dict['x_3d'].F = x_3d
        else:
            mm_data_dict['x_3d'] = x_3d
        mm_data_dict['modalities'][modality] = mod_data
        if mm_data_dict['x_seen'] is None:
            mm_data_dict['x_seen'] = x_seen
        else:
            mm_data_dict['x_seen'] = torch.logical_or(
                x_seen, mm_data_dict['x_seen'])

        return mm_data_dict

    def extra_repr(self) -> str:
        repr_attr = ['drop_3d', 'drop_mod', 'keep_last_view', 'checkpointing']
        return "\n".join([f'{a}={getattr(self, a)}' for a in repr_attr])


class IdentityBranch(BaseModule):
    def __init__(self):
        super(IdentityBranch, self).__init__()

    def forward(self, mm_data_dict, modality):
        return mm_data_dict
