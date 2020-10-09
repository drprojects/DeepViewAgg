import numpy as np
from torch_geometric.data import Data
from torch_points3d.datasets.multimodal.image import ImageData
from .projection import compute_index_map



class PointImagePixelMapping:
    """
    Transform-like structure. Intended to be called on _datas and images 
    poses.
    """
    def __init__(
            self,
            img_mask=None,
            img_size=(1024, 512),
            crop_top=0,
            crop_bottom=0,
            voxel=0.1,
            r_max=30,
            r_min=0.5,
            growth_k=0.2,
            growth_r=10,
            empty=0):

        self.img_size = tuple(img_size)
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
        self.voxel = voxel
        self.r_max = r_max
        self.r_min = r_min
        self.growth_k = growth_k
        self.growth_r = growth_r
        self.empty = empty

        if img_mask:
            img_mask = np.asarray(img_mask)
            assert (img_mask.shape == self.img_size, 
                f"Mask shape {img_mask.shape} does not match image size {self.img_size}.")
        self.img_mask = img_mask


    def _process(self, data, images):
        
        assert isinstance(images[0], ImageData)

        ########################################################################
        # project indices with compute_index_map
        # invert from image-based indexing to point-based
        # keep result in attributes
        ########################################################################

        return data


    def __call__(self, data, images):
        """
        Compute the projection of data points into images and return the input 
        data augmented with attributes mapping points to pixels in provided 
        images.

        Expects a Data and a List(ImageData) or a List(Data) and a List(List(ImageData)) of 
        matching length.
        """
        assert isinstance(images, list)

        if isinstance(data, list):
            assert len(data) == len(images), (f"List(Data) items and List(List(ImageData)) must ",
                "have the same lengths.")
            data = [self._process(d, i) for d, i in zip(data, images)]

        else:
            data = self._process(data, images)

        return data


    def __repr__(self):
        return self.__class__.__name__

#-------------------------------------------------------------------------------

class PointImagePixelMappingFromId:
    """
    Transform-like structure. Intended to be called on _datas and _images_datas
    and images poses.

    Populate data sample in place with image attributes in data_multimodal, 
    based on the self.key point identifiers. The indices in data are expected 
    to be included in those in data_multimodal.
    """
    def __init__(self, key='processed_id'):
        self.key = key


    def _process(self, data, data_multimodal):
        assert hasattr(data, self.key)

        ########################################################################
        # recover mappings and images
        # stack them carefully, taking care of the indices
        ########################################################################

        return data


    def __call__(self, data, data_multimodal):
        """
        Populate data sample in place with image attributes in data_multimodal,
        based on the self.key point identifiers.
        """
        if isinstance(data, list):
            data = [self._process(d, data_multimodal) for d in data]
        else:
            data = self._process(data, data_multimodal)
        return data


    def __repr__(self):
        return self.__class__.__name__
