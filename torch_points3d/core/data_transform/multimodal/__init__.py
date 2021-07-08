import sys

from .compose import ComposeMultiModal
from .image import *

_custom_multimodal_transforms = sys.modules[__name__]


def instantiate_multimodal_transform(transform_option, attr="transform"):
    """
    Creates a multimodal transform from an OmegaConf dict. Inspired from such as:
    transform: GridSampling3D
        params:
            size: 0.01
    """
    tr_name = getattr(transform_option, attr, None)
    tr_params = getattr(transform_option, 'params', None)
    cls = getattr(_custom_multimodal_transforms, tr_name, None)

    if not cls:
        raise ValueError(f"Multimodal transform {tr_name} is nowhere to be found")

    if tr_params:
        return cls(**tr_params)

    return cls()


def instantiate_multimodal_transforms(transform_options):
    """ Creates a torch_geometric composite transform from an OmegaConf list such as
    - transform: GridSampling3D
        params:
            size: 0.01
    - transform: NormaliseScale
    """
    transforms = []
    for transform in transform_options:
        transforms.append(instantiate_multimodal_transform(transform))
    return ComposeMultiModal(transforms)
