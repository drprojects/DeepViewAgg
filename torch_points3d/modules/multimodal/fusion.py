from abc import ABC

import torch
import torch.nn as nn


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

    MODES = ['residual', 'concatenation', 'both', 'modality']

    def __init__(self, mode='residual', **kwargs):
        super(BimodalFusion, self).__init__()
        self.mode = mode
        if self.mode == 'residual':
            self.f = lambda a, b: a + b
        elif self.mode == 'concatenation':
            self.f = lambda a, b: torch.cat((a, b), dim=-1)
        elif self.mode == 'both':
            self.f = lambda a, b: torch.cat((a, a + b), dim=-1)
        elif self.mode == 'modality':
            self.f = lambda a, b: b
        else:
            raise NotImplementedError(
                f"Unknown fusion mode='{mode}'. Please choose among "
                f"supported modes: {self.MODES}.")

    def forward(self, x_main, x_mod):
        if x_main is None:
            return x_mod
        if x_mod is None:
            return x_main

        # If the x_mod is a sparse tensor, we only keep its features
        x_mod = x_mod if isinstance(x_mod, torch.Tensor) else x_mod.F

        # Update the x_main while respecting its format
        x_main = self.f(x_main, x_mod)

        return x_main

    def extra_repr(self) -> str:
        return f"mode={self.mode}"
