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
        self.down_block = down_block if down_block is not None else Identity
        self.conv_block = conv_block if conv_block is not None else Identity

        # Initialize the dict holding the conv and merge blocks for all modalities
        self.modality_blocks = {}
        self._init_from_kwargs(**kwargs)

        # Expose the 3D down_conv .sampler attribute (for UnwrappedUnetBasedModel)
        # TODO this is for KPConv, is it doing the intended, is it needed at all ?
        self.sampler = [getattr(self.down_block, "sampler", None),
                        getattr(self.conv_block, "sampler", None)]

        # TODO : check layers compatibility

    def _init_from_kwargs(self, **kwargs):
        """Kwargs are expected to carry fully-fledged modality-specific conv
        and merge modules in the following format:
            kwargs[modality] = {'conv': conv_block, 'merge': merge_block}.
        """
        for m in MODALITY_NAMES:
            if m in kwargs.keys():
                if 'conv' not in kwargs[m].keys():
                    raise ValueError(f"Modality '{m}' requires a 'conv' module.")
                elif 'merge' not in kwargs[m].keys():
                    raise ValueError(f"Modality '{m}' requires a 'merge' module.")
                else:
                    self.modality_blocks[m] = kwargs[m]

    @property
    def modalities(self):
        return list(self.modality_blocks.keys())

    @property
    def num_modalities(self):
        return len(self.modalities) + 1

    def extra_repr(self):
        return f"(modalities): {tuple(self.modalities)}"

    def forward(self, mm_data_tuple):
        """
        Forward pass of the MultiModalBlockDown.

        Expects a tuple of 3D data (Data, SparseTensor, etc.) destined
        for the 3D convolutional modules, and a dictionary of
        modality-specific data equipped with corresponding mappings.
        """
        # Unpack the multimodal data tuple
        x_3d, mod_dict = mm_data_tuple

        print("3D conv down...")
        # Conv on the main 3D modality - assumed to reduce 3D resolution
        x_3d, idx, idx_mode = self.forward_3d_block_down(self.down_block, x_3d)

        for m in self.modalities:
            print(f"{m} mapping 3D downsampling with '{idx_mode}' mode...")
            # Update modality data and mappings wrt point idx
            mod_dict[m] = mod_dict[m].select_points(idx, mode=idx_mode)

            print(f"{m} conv down...")
            # Conv on the modality data. The modality data holder
            # carries a feature tensor per modality settings. Hence the
            # modality features are provided as a list of tensors.
            # TODO: this is multisetting-oriented: list of feature
            #  tensors Need to make the multisetting the default, for
            #  all modalities
            x_mod = [self.modality_blocks[m]['conv'](x) for x in mod_dict[m].x]

            print(f"{m} mapping modality downsampling...")
            # Update modality features and mappings wrt modality scale.
            # Note that convolved features are preserved in the modality
            # data holder, to be later used in potential downstream
            # modules.
            mod_dict[m] = mod_dict[m].update_features_and_scale(x_mod)

            # Extract CSR-arranged atomic features from the feature maps
            # of each input modality setting
            x_mod = [x[idx]
                     for x, idx
                     in zip(x_mod, mod_dict[m].feature_map_indexing)]

            # Atomic pooling of the modality features on each
            # separate setting
            x_mod = [self.modality_blocks[m]['atomic'](x, x_3d, a_idx)
                     for x, a_idx
                     in zip(x_mod, mod_dict[m].atomic_csr_indexing)]

            # Concatenate view-level features from each input
            # modality setting
            x_mod = torch.cat(x_mod, dim=0)

            # Sort the concatenated view-level features to a
            # CSR-friendly order wrt 3D points features
            x_mod = x_mod[mod_dict[m].view_cat_sorting]

            # View pooling of the joint modality features
            v_idx = mod_dict[m].view_cat_csr_indexing
            x_mod = self.modality_blocks[m]['view'](x_mod, x_3d, v_idx)

            # Fuse the modality features into the 3D points features
            x_3d = self.modality_blocks[m]['fuse'](x_mod, x_3d)

        # Conv on the main 3D modality
        x_3d = self.conv_block(x_3d)

        return tuple((x_3d, mod_dict))

    @staticmethod
    def forward_3d_block_down(block, x):
        """
        Wrapper method to apply the forward pass on a 3D down conv
        block.

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
        # MinkowskiEngine forward and reindexing
        if isinstance(x, me.SparseTensor):
            mode = 'merge'

            # Forward pass on the block while keeping track of the
            # stride levels
            stride_in = x.tensor_stride[0]
            x = block(x)
            stride_out = x.tensor_stride[0]

            if stride_in == stride_out:
                return x, None, mode

            src, target = x.coords_man.get_coords_map(stride_in, stride_out)
            idx = target[src.argsort()]
            return x, idx, mode

        # TorchSparse forward and reindexing
        elif isinstance(x, ts.SparseTensor):
            mode = 'merge'

            # Forward pass on the block while keeping track of the
            # stride levels
            stride_in = x.s
            x = block(x)
            stride_out = x.s

            if stride_in == stride_out:
                return x, None, mode

            if stride_out % stride_in == 0:
                ratio = int(stride_out / stride_in)
            else:
                raise ValueError(
                    f"Output stride {stride_out} should be a multiple of input "
                    f"stride {stride_in}.")
            in_coords = x.coord_maps[stride_in] // ratio * ratio
            out_coords = x.coord_maps[stride_out]
            idx = sphashquery(sphash(in_coords), sphash(out_coords))
            return x, idx, mode

        # Non-sparse forward and reindexing
        else:
            mode = 'pick'

            # Forward pass on the block while keeping track of the
            # sampler indices
            block.sampler.last_idx = None
            x = block(x)
            idx = block.sampler.last_idx
            return x, idx, mode
