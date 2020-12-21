import torch.nn as nn
from .pooling import *
from .fusion import BimodalFusion
from .pooling import BimodalPool


class BimodalMerge(nn.Module):
    """Bimodal
    Input
        X - 3D features [BxN, Cin]
        Xmod - features from modality
        Mmod - mappings from modality

    Output
        X - merged features [BxN, Cout]
    """
    def __init__(self, fusion='residual', aggregation=None, **kwargs):
        # TODO : parse opt to get the aggregation and fusion parameters
        self.fusion = BimodalFusion(fusion)
        self.aggregation = aggregation


    def forward(self, mm_data):
        raise NotImplementedError
