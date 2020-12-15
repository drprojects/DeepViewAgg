import logging
import torch.nn.functional as F
import torch.nn as nn
import torchsparse as TS


from torch_points3d.models.base_model import BaseModel
from torch_points3d.datasets.segmentation import IGNORE_LABEL
# from torch_points3d.applications.sparseconv3d import SparseConv3d
from torch_points3d.models.base_architectures.unet import UnwrappedUnetBasedModel


log = logging.getLogger(__name__)


class MMUnet(UnwrappedUnetBasedModel):
    pass
