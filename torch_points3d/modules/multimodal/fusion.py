import torch
import torch.nn as nn


class BimodalFusion(nn.Module):

    MODES = ['residual', 'concatenation']

    def __init__(self, mode='residual', **kwargs):
        super().__init__()
        assert mode in self.MODES, 'Unknown fusion mode: {mode}'
        self.mode = mode

    def forward(self, X_main, X_mod):
        if self.mode == 'residual':
            return X_main + X_mod
        elif self.mode == 'concatenation':
            return torch.cat((X_main, X_mod), dim=-1)
