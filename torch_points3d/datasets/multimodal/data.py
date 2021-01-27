import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch_points3d.datasets.multimodal.image import *
from torch_points3d.core.data_transform.multimodal.image import \
    SelectMappingFromPointId, _MAPPING_KEY
from torch_points3d.utils.multimodal import tensor_idx

MODALITY_NAMES = ["image"]


class MMData(object):
    """
    A holder for multimodal data.

    Combines 3D point in torch_geometric Data, Images in ImageData and
    mappings in CSRData objects.

    Provides sanity checks to ensure the validity of the data, along
    with loading methods to leverage multimodal information with
    Pytorch.
    """

    def __init__(self, data, images):
        self.data = data
        self.images = images
        self.key = _MAPPING_KEY
        self.debug()

    def debug(self):
        assert isinstance(self.data, Data)
        assert isinstance(self.images, (ImageData, MultiSettingImageData))
        assert self.images.num_points > 0

        # Ensure Data have the key attribute necessary for linking
        # points with images in mappings. Each point must have a
        # mapping, even if empty.
        # NB: just like images, the same point may be used multiple
        #  times.
        assert hasattr(self.data, self.key)
        assert 'index' in self.key, \
            f"Key {self.key} must contain 'index' to benefit from " \
            f"Batch mechanisms."
        idx = np.unique(self.data[self.key])
        assert idx.max() + 1 == idx.shape[0] == self.num_points \
               == self.images.num_points, \
            f"Discrepancy between the Data point indices and the mappings. " \
            f"Data {self.key} counts {idx.shape[0]} unique values with " \
            f"max={idx.max()}, with {self.num_points} points in total, while " \
            f"mappings cover point indices in [0, {self.images.num_points}]."

    def __len__(self):
        return self.data.num_nodes

    @property
    def num_points(self):
        return self.data.num_nodes

    @property
    def num_images(self):
        return self.images.num_images

    @property
    def num_node_features(self):
        return self.data.num_node_features

    def to(self, device):
        self.data = self.data.to(device)
        self.images = self.images.to(device)
        return self

    @property
    def device(self):
        return self.images.device

    def load_images(self):
        self.images.load_images()
        return self

    def clone(self):
        return MMData(
            self.data.clone(),
            self.images.clone())

    def __getitem__(self, idx):
        """
        Indexing mechanism on the points.

        Returns a new copy of the indexed MMData, with updated ImageData
        and ImageMapping. Supports torch and numpy indexing.
        """
        idx = tensor_idx(idx)
        idx = idx.to(self.device)

        # Index the Data first
        data = self.data.clone()
        for key, item in self.data:
            if torch.is_tensor(item) and item.size(0) == self.data.num_nodes:
                data[key] = data[key][idx]

        # Update the ImageData and ImageMapping accordingly
        # Data and ImageData should be cloned beforehand because
        # transforms may affect the input parameters in-place
        transform = SelectMappingFromPointId()
        data, images = transform(data, self.images)

        return MMData(data, images)

    def __repr__(self):
        info = [f"    {key} = {getattr(self, key)}"
                for key in ['data', 'images']]
        info = '\n'.join(info)
        return f"{self.__class__.__name__}(\n{info}\n)"


class MMBatch(MMData):
    """
    A wrapper around MMData to create batches of multimodal data while
    leveraging the batch mechanisms for each modality attribute.

    Relies on several assumptions that MMData.debug() keeps in check. 
    """

    def __init__(self, data, images):
        super(MMBatch, self).__init__(data, images)
        self.__sizes__ = None

    @property
    def batch_pointers(self):
        return np.cumsum(np.concatenate(([0], self.__sizes__))) \
            if self.__sizes__ is not None \
            else None

    @property
    def batch_items_sizes(self):
        return self.__sizes__ if self.__sizes__ is not None else None

    @property
    def num_batch_items(self):
        return len(self.__sizes__) if self.__sizes__ is not None \
            else None

    def clone(self):
        out = MMBatch(
            self.data.clone(),
            self.images.clone())
        out.__sizes__ = self.__sizes__
        return out

    @staticmethod
    def from_mm_data_list(mm_data_list):
        assert isinstance(mm_data_list, list) and len(mm_data_list) > 0
        assert all([isinstance(mm_data, MMData) for mm_data in mm_data_list])

        data = Batch.from_data_list(
            [mm_data.data for mm_data in mm_data_list])
        im_batch_class = MultiSettingImageBatch \
            if isinstance(mm_data_list[0].images, MultiSettingImageData) \
            else ImageBatch
        images = im_batch_class.from_image_data_list(
            [mm_data.images for mm_data in mm_data_list])
        sizes = [len(mm_data) for mm_data in mm_data_list]

        batch = MMBatch(data, images)
        batch.__sizes__ = np.array(sizes)

        return batch

    def to_mm_data_list(self):
        if self.__sizes__ is None:
            raise RuntimeError(
                'Cannot reconstruct multimodal data list from batch '
                'because the batch object was not created using '
                '`MMBatch.from_mm_data_list()`.')

        data_list = self.data.to_data_list()
        images_list = self.images.to_image_data_list()

        return [MMData(data, images)
                for data, images
                in zip(data_list, images_list)]
