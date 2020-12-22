import torch
import torch.nn as nn


class BimodalFusion(nn.Module):

    MODES = ['residual', 'concatenation']

    def __init__(self, mode='residual', **kwargs):
        super().__init__()
        assert mode in self.MODES, 'Unknown fusion mode: {mode}'
        self.mode = mode

    def forward(self, x_main, x_mod):
        if self.mode == 'residual':
            return x_main + x_mod
        elif self.mode == 'concatenation':
            return torch.cat((x_main, x_mod), dim=-1)
