import numpy as np
from torch_geometric.data import Data
from torch_points3d.datasets.multimodal.image import ImageData
from torch_points3d.datasets.multimodal.forward_star import ForwardStar
from .projection import compute_index_map



class ComposeMultiModal(object):
    """Composes several multimodal transforms together.
    Args:
        transforms (list of :obj:`transform` objects): List of transforms to
            compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args

    def __repr__(self):
        args = ['    {},'.format(t) for t in self.transforms]
        return '{}([\n{}\n])'.format(self.__class__.__name__, '\n'.join(args))