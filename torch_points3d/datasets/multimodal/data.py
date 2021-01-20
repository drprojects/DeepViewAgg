import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch_points3d.datasets.multimodal.image import *
from torch_points3d.datasets.multimodal.csr import CSRData, CSRBatch
from torch_points3d.core.data_transform.multimodal.image import \
    SelectMappingFromPointId

MODALITY_NAMES = ["image"]


# TODO : modify MMData to fit the structure in torch_points3d/modules/multimodal/modules.py

class MMData(object):
    """
    A holder for multimodal data.

    Combines 3D point in torch_geometric Data, Images in ImageData and
    mappings in CSRData objects.

    Provides sanity checks to ensure the validity of the data, along
    with loading methods to leverage multimodal information with
    Pytorch.
    """

    def __init__(self, data, images, key='point_index'):
        self.data = data
        self.images = images
        self.key = key
        self.debug()

    def debug(self):
        # TODO: this is ImageData-centric. Need to extend to other modalities
        assert isinstance(self.data, Data)
        assert isinstance(self.images, (ImageData, ImageDataList))
        assert self.images.mappings is not None

        # Ensure Data have the key attribute necessary for linking
        # points with images in mappings. Each point must have a
        # mapping, even if empty.
        # NB: just like images, the same point may be used multiple
        #  times.
        assert hasattr(self.data, self.key)
        assert 'index' in self.key, \
            f"Key {self.key} must contain 'index' to benefit from " \
            f"Batch mechanisms."
        assert np.array_equal(np.unique(self.data[self.key]),
                              np.arange(len(self.images.mappings))), \
            "Data point indices must span the entire range of mappings."

    def __len__(self):
        return self.data.num_nodes

    @property
    def num_points(self):
        return self.data.num_nodes

    @property
    def num_images(self):
        return self.images.num_images

    def to(self, device):
        self.data = self.data.to(device)
        self.images = self.images.to(device)
        return self

    @property
    def device(self):
        return self.images.device

    def load_images(self):
        self.images.load_images()

    def clone(self):
        return MMData(
            self.data.clone(),
            self.images.clone(),
            key=self.key)

    def __getitem__(self, idx):
        """
        Indexing mechanism on the points.

        Returns a new copy of the indexed MMData, with updated ImageData
        and ImageMapping. Supports torch and numpy indexing.
        """
        if isinstance(idx, int):
            idx = torch.LongTensor([idx])
        elif isinstance(idx, list):
            idx = torch.LongTensor(idx)
        elif isinstance(idx, slice):
            idx = torch.arange(self.num_points)[idx]
        elif isinstance(idx, np.ndarray):
            idx = torch.from_numpy(idx)
        # elif not isinstance(idx, torch.LongTensor):
        #     raise NotImplementedError
        if isinstance(idx, torch.BoolTensor):
            idx = torch.where(idx)[0]
        assert idx.dtype is torch.int64, \
            "MMData only supports int and torch.LongTensor indexing."
        assert idx.shape[0] > 0, \
            "MMData only supports non-empty indexing. At least one " \
            "index must be provided."
        idx = idx.to(self.device)

        # Index the Data first
        data = self.data.clone()
        for key, item in self.data:
            if torch.is_tensor(item) and item.size(0) == self.data.num_nodes:
                data[key] = data[key][idx]

        # Update the ImageData and ImageMapping accordingly
        transform = SelectMappingFromPointId(key=self.key)
        data, images = transform(data, self.images)

        return MMData(data, images, key=self.key)

    def __repr__(self):
        info = [f"    {key} = {getattr(self, key)}"
                for key in ['data', 'images']]
        info = '\n'.join(info)
        return f"{self.__class__.__name__}(\n{info}\n)"


# TODO : modify MMBatch to fit the structure in torch_points3d/modules/multimodal/modules.py

class MMBatch(MMData):
    """
    A wrapper around MMData to create batches of multimodal data while
    leveraging the batch mechanisms for each modality attribute.

    Relies on several assumptions that MMData.debug() keeps in check. 
    """

    def __init__(self, data, images, key='point_index'):
        super(MMBatch, self).__init__(data, images, key=key)
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
            self.images.clone(),
            key=self.key)
        out.__sizes__ = self.__sizes__
        return out

    @staticmethod
    def from_mm_data_list(mm_data_list, key='point_index'):
        assert isinstance(mm_data_list, list) and len(mm_data_list) > 0
        assert all([isinstance(mm_data, MMData) for mm_data in mm_data_list])

        data = Batch.from_data_list(
            [mm_data.data for mm_data in mm_data_list])
        images = ImageBatch.from_image_data_list(
            [mm_data.images for mm_data in mm_data_list])
        sizes = [len(mm_data) for mm_data in mm_data_list]

        batch = MMBatch(data, images, key=key)
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

        return [MMData(data, images, key=self.key)
                for data, images
                in zip(data_list, images_list)]
