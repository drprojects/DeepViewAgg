import torch.nn as nn
from torch_points3d.datasets.multimodal.data import MODALITY_NAMES
from torch_points3d.core.common_modules.base_modules import Identity


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
        # TODO get the 3D conv type and set input from the down module ?
        # TODO get the 3D sampling from the down module ?

        # Initialize the dict holding the conv and merge blocks for all modalities
        self.modality_blocks = {}
        self._init_from_kwargs(**kwargs)

        # TODO : create modality-specific block modules
        # TODO : create merge block modules

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
        """Forward pass of the MultiModalBlockDown.

        Expects a tuple of 3D data (Data, SparseTensor, etc.) destined
        for the 3D convolutional modules, and a dictionary of
        modality-specific data equipped with corresponding mappings.
        """
        # Unpack the multimodal data tuple
        data_3d, data_mod = mm_data_tuple

        # TODO : what about SparseTensor and other weird formats ?
        # Conv on the main 3D modality - assumed to reduce 3D resolution
        print('3D conv down...')
        data_3d = self.down_block(data_3d)

        for m in self.modalities:
            # Update mappings after the 3D down conv
            # TODO : recover sampling indices and update mappings based on the
            #  3D downconv. KpConv uses GridSampling3D, PointNet++ uses FPS
            #  sampler, SparseConv uses strides, ... Not uniform. Should store
            #  sampling idx in the conv module after sampling ?
            #  FOR NON-SPARSE CONV, save the last resampling in self.sampler...
            #  hoping there are never 2 samplings ? Or update samplings when
            #  called and reset it when required (here) ?
            #  NORMALLY, if I am correct, even the MultiScale convs sample only
            #  once (it is the neighborhood search they do multiple times).
            print(f'{m} mapping 3D downsampling...')

            def get_3d_sampling_idx(data, down_conv):
                # if data is sparse, get mapping from here
                # else, get last_idx from conv sampler
                # otherwise raise error
                return 
            idx = self.down_block.sampler.last_idx

            data_mod[m].mappings = update_mappings(data_mod[m])

            print(f'{m} conv down...')
            # Conv on the modality-specific data
            data_mod[m].data = self.modality_blocks[m]['conv'](data_mod[m])

            print(f'{m} mapping modality downsampling...')
            # Update mappings after modality conv
            # TODO : update mappings after modality conv
            data_mod[m].mappings = update_mappings(data_mod[m])

            # Merge the modality into the main 3D features
            # TODO : create merge blocks class. Are they modality-specific ?
            data_3d = self.modality_blocks[m]['merge'](data_3d, data_mod[m])

        # Conv on the main 3D modality
        data_3d = self.conv_block(data_3d)

        return (data_3d, data_mod)
