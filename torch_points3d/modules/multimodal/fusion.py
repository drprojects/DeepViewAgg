import torch
import torch.nn as nn


class BimodalFusion(nn.Module):
    """Bimodal pooling combines pooled features from a modality with features
    from the main modality.

    The main and additional modalities' feature tensors are expected to have
    sizes [N x C_main] and [N x C_mod] matching. For residual fusion, we
    further require C_mod = C_main.
    """

    MODES = ['residual', 'concatenation']

    def __init__(self, mode='residual', **kwargs):
        super(BimodalFusion, self).__init__()
        assert mode in self.MODES, 'Unknown fusion mode: {mode}'
        self.mode = mode

    def forward(self, x_main, x_mod):
        if self.mode == 'residual':
            return x_main + x_mod
        elif self.mode == 'concatenation':
            return torch.cat((x_main, x_mod), dim=-1)
