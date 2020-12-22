import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch_points3d.datasets.multimodal.image import ImageData, ImageBatch
from torch_points3d.datasets.multimodal.csr import CSRData, CSRDataBatch

MODALITY_NAMES = ["image"]


# TODO : modify MMData to fit the structure in torch_points3d/modules/multimodal/modules.py

class MMData(object):
    """
    A holder for multimodal data.

    Combines 3D point in torch_geometric Data, Images in ImageData and mappings 
    in CSRData objects.

    Provides sanity checks to ensure the validity of the data, along with
    loading methods to leverage multimodal information with Pytorch.    
    """

    def __init__(self, data, images, mappings, key='point_index'):
        self.data = data
        self.images = images
        self.mappings = mappings
        self.key = key
        self.debug()

    def debug(self):
        assert isinstance(self.data, Data)
        assert isinstance(self.images, ImageData)
        assert isinstance(self.mappings, CSRData)

        # Ensure Data have the key attribute necessary for linking points with
        # images in mappings. Each point must have a mapping, even if empty.
        # NB: just like images, the same point may be used multiple times.
        assert hasattr(self.data, self.key)
        assert 'index' in self.key, \
            f"Key {self.key} must contain 'index' to benefit from Batch mechanisms."
        assert np.array_equal(np.unique(self.data[self.key]), np.arange(len(self.mappings))), \
            "Data point indices must span the entire range of mappings."

        # Ensure mappings have the expected signature
        self.mappings.debug()
        assert self.mappings.num_values == 2 \
               and self.mappings.is_index_value[0] \
               and isinstance(self.mappings.values[1], CSRData), \
            "Mappings must have the signature of PointImagePixels mappings."

        # Ensure all images in ImageData are used in the mappings.
        # Otherwise, some indexing errors may arise when batching.
        # In fact, we would only need to ensure that the largest image index in 
        # the mappings corresponds to the number of images, but this is safer
        # and avoids loading unnecessary ImageData.
        assert np.array_equal(np.unique(self.mappings.values[0]),
                              np.arange(self.images.num_images)), \
            "Mapping image indices must span the entire range of images."

        # Ensure pixel coordinates in the mappings are compatible with 
        # the expected feature maps resolution.
        pix_max = self.mappings.values[1].values[0].max(axis=0)
        map_max = self.images.map_size_low
        assert all(a < b for a, b in zip(pix_max, map_max)), \
            "Pixel coordinates must match images.map_size_low."

    def __len__(self):
        return self.data.num_nodes

    def to(self, device):
        # TODO pass mappings to device too
        self.images = self.images.to(device)
        self.data = self.data.to(device)
        return self

    @property
    def device(self):
        return self.images.device

    def load_images(self, idx=None, device=None, size=None):
        if device is None:
            device = self.device
        return self.images.read_images(idx=idx, size=size).to(device)

    def __repr__(self):
        info = [f"    {key} = {getattr(self, key)}" for key in ['data', 'images', 'mappings']]
        info = '\n'.join(info)
        return f"{self.__class__.__name__}(\n{info}\n)"


# TODO : modify MMBatch to fit the structure in torch_points3d/modules/multimodal/modules.py

class MMBatch(MMData):
    """
    A wrapper around MMData to create batches of multimodal data while
    leveraging the batch mechanisms for each modality attribute.

    Relies on several assumptions that MMData.debug() keeps in check. 
    """

    def __init__(self, data, images, mappings, key='point_index'):
        super(MMBatch, self).__init__(data, images, mappings, key=key)
        self.__sizes__ = None

    @property
    def batch_jumps(self):
        return np.cumsum(np.concatenate(([0], self.__sizes__))) if self.__sizes__ is not None \
            else None

    @property
    def batch_items_sizes(self):
        return self.__sizes__ if self.__sizes__ is not None else None

    @property
    def num_batch_items(self):
        return len(self.__sizes__) if self.__sizes__ is not None else None

    @staticmethod
    def from_mm_data_list(mm_data_list, key='point_index'):
        assert isinstance(mm_data_list, list) and len(mm_data_list) > 0
        assert all([isinstance(mm_data, MMData) for mm_data in mm_data_list])

        data = Batch.from_data_list([mm_data.data for mm_data in mm_data_list])
        images = ImageBatch.from_image_data_list([mm_data.images for mm_data in mm_data_list])
        mappings = CSRDataBatch.from_csr_list([mm_data.mappings for mm_data in mm_data_list])
        sizes = [len(mm_data) for mm_data in mm_data_list]

        batch = MMBatch(data, images, mappings, key=key)
        batch.__sizes__ = np.array(sizes)

        return batch

    def to_mm_data_list(self):
        if self.__sizes__ is None:
            raise RuntimeError(('Cannot reconstruct multimodal data list from batch because the ',
                                'batch object was not created using `MMBatch.from_mm_data_list()`.'))

        data_list = self.data.to_data_list()
        images_list = self.images.to_image_data_list()
        mappings_list = self.mappings.to_csr_list()

        return [MMData(data, images, mappings, key=self.key)
                for data, images, mappings
                in zip(data_list, images_list, mappings_list)]
