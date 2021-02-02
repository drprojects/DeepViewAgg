from abc import ABC

import torch
import torch.nn as nn
import MinkowskiEngine as me
import torchsparse as ts


class BimodalFusion(nn.Module, ABC):
    """Bimodal fusion combines features from different modalities into
    a single tensor.

    The input modalities' feature tensors are expected to have matching
    sizes [N x C_1] and [N x C_2]. For residual fusion, we further
    require C_1 = C_2.

    By convention, the second features are fused into the first, main
    modality. This matters as the output format will match that of the
    main modality
    """

    MODES = ['residual', 'concatenation']

    def __init__(self, mode='residual', **kwargs):
        super(BimodalFusion, self).__init__()
        assert mode in self.MODES, 'Unknown fusion mode: {mode}'
        self.mode = mode
        if self.mode == 'residual':
            self.f = lambda a, b: a + b
        elif self.mode == 'concatenation':
            self.f = lambda a, b: torch.cat((a, b), dim=-1)
        else:
            raise NotImplementedError

    def forward(self, x_main, x_mod):
        if x_main is None:
            return x_mod
        if x_mod is None:
            return x_main

        # If the x_mod is a sparse tensor, we only keep its features
        x_mod = x_mod.F \
            if isinstance(x_mod, (me.SparseTensor, ts.SparseTensor)) \
            else x_mod

        # Update the x_main while respecting its format
        if isinstance(x_main, (me.SparseTensor, ts.SparseTensor)):
            x_main.F = self.f(x_main.F, x_mod)
        else:
            x_main = self.f(x_main, x_mod)
        return x_main
