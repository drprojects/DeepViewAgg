import torch.nn as nn
from .pooling import *
from .fusion import BimodalFusion
from .pooling import BimodalPool


class BimodalMerge(nn.Module):
    """Bimodal pooling followed by bimodal fusion.
    Input
        X - 3D features [BxN, Cin]
        Xmod - features from modality
        Mmod - mappings from modality

    Output
        X - merged features [BxN, Cout]

    IMPORTANT: the order of 3D points in the main modality is expected to
    match that of the indexes in the mappings. Any update of the mappings
    following a reindexing, reordering or sampling of the 3D points must be
    performed prior to the multimodal pooling.
    """
    def __init__(self, **kwargs):
        self.pooling = BimodalPool(**kwargs)
        self.fusion = BimodalFusion(**kwargs)

    def forward(self, x_main, x_mod, mappings):
        x_agg = self.poling(x_main, x_mod, mappings)
        x_main = self.fusion(x_main, x_agg)
        return x_main, x_mod, mappings
