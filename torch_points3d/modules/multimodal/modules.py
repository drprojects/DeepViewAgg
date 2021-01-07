import torch.nn as nn
from torch_points3d.datasets.multimodal.data import MODALITY_NAMES
from torch_points3d.core.common_modules.base_modules import Identity


class MultimodalBlockDown(nn.Module):
    """Multimodal block with downsampling that looks like:

    IN MMData         --- 3D Down Conv ---- Merge 1 --- Merge i ---- 3D Conv ---         OUT MMData
                                              |           |
                      --- Mod 1 Down Conv ----           |
                                                        |
                      ...                              |
                                                      |
                      --- Mod i Down Conv ------------

                      ...
    """
    def __init__(self, down_block, conv_block, **kwargs):
        """Build the Multimodal module from already-instantiated modules.
        Modality-specific modules are expected to be passed in dictionaries
        holding the conv and merge modules under 'conv' and 'merge' keys.
        """
        # BaseModule initialization
        super().__init__()

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
        self.sampler = [getattr(self.down_block, "sampler", None), getattr(self.down_block, "sampler", None)]

        # TODO : check layers compatibility

    def _init_from_kwargs(self, **kwargs):
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

    def forward(self, mm_data):
        # TODO : what about SparseTensor and other weird formats ?
        # Conv on the main 3D modality - assumed to reduce 3D resolution
        mm_data.data = self.down_block(mm_data.data)

        for m in self.modalities:
            # Update mappings after the 3D down conv
            # TODO : recover sampling indices and update mappings based on the 3D downconv. KpConv uses GridSampling3D,
            #  PointNet++ uses FPS sampler, SparseConv uses strides, ... Not uniform. Should store sampling idx in the
            #  conv module after sampling ?
            mm_data.modalities[m].mappings = update_mappings(mm_data)

            # Conv on the modality-specific data
            mm_data.modalities[m].data = self.modality_blocks[m]['conv'](mm_data.modalities[m].data)

            # Update mappings after modality conv
            # TODO : update mappings after modality conv
            mm_data.modalities[m].mappings = update_mappings(mm_data)

            # Merge the modality into the main 3D features
            # TODO : create merge blocks class. Are they modality-specific ?
            mm_data.data = self.merge_block_mod(
                mm_data.data,
                mm_data.modalities[m].data,
                mm_data.modalities[m].mappings
            )

        # Conv on the main 3D modality
        mm_data.data = self.conv_block(mm_data.data)
        return mm_data
