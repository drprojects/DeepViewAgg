import torch
import torch.nn as nn


class ModalityDropout(nn.Module):

    def __init__(self, p=0):
        super(ModalityDropout, self).__init__()
        assert 0 <= p <= 1, f'p must be in [0, 1].'
        self.p = p

    def forward(self, x):
        if not self.training:
            return x.mul(1 / (1 - self.p))
        return x.mul(torch.rand(1) > self.p)
