import torch
import numpy as np
from torch_geometric.data import Data, Batch
from torch_points3d.core.multimodal.image import ImageData
from torch_points3d.utils.multimodal import tensor_idx, MAPPING_KEY


# Supported modalities
MODALITY_FORMATS = {"image": ImageData}
MODALITY_NAMES = list(MODALITY_FORMATS.keys())


class MMData(object):
    """
    A holder for multimodal data.

    Combines 3D point in torch_geometric Data, with modality-specific
    data representations equipped with mappings to the 3D Data points.
    These modalities are expected to be passed as kwargs, with keywords
    matching supported modalities in MODALITIES.

    Provides sanity checks to ensure the validity of the data, along
    with loading methods to leverage multimodal information with
    Pytorch.
    """

    def __init__(self, data: Data, **kwargs):
        self.data = data
        self.modalities = kwargs
        self.mapping_key = MAPPING_KEY
        for k in self.data.keys:
            setattr(self, k, getattr(self.data, k))
        self.debug()

    def debug(self):
        assert isinstance(self.data, Data)

        # Ensure Data have the key attribute necessary for linking
        # points with modality mappings. Each point must have a
        # mapping, even if empty.
        # NB: just like images, the same point may be used multiple
        #  times.
        assert hasattr(self.data, self.mapping_key)
        assert 'index' in self.mapping_key, \
            f"Key {self.mapping_key} must contain 'index' to benefit from " \
            f"Batch mechanisms."
        idx = torch.unique(self.data[self.mapping_key])
        assert idx.max() + 1 == idx.shape[0] == self.num_points, \
            f"Discrepancy between the Data point indices and the mappings " \
            f"indices. Data {self.mapping_key} counts {idx.shape[0]} unique " \
            f"values with max={idx.max()}, with {self.num_points} points in " \
            f"total."

        # Modality-specific checks
        for mod, data_mod in self.modalities.items():
            assert mod in MODALITY_NAMES, \
                f"Received kwarg={mod} but expected key to belong to " \
                f"supported modalities: {MODALITY_NAMES}."

            assert isinstance(data_mod, MODALITY_FORMATS[mod]), \
                f"Expected modality '{mod}' data to be of type " \
                f"{MODALITY_FORMATS[mod]} but got type {type(data_mod)} " \
                f"instead."
            # assert data_mod.num_points > 0
            assert self.num_points == data_mod.num_points \
                   or data_mod.num_points == 0, \
                f"Discrepancy between the Data point indices and the '{mod}' " \
                f"modality mappings. Data '{self.mapping_key}' counts " \
                f"{self.num_points} points in total, while '{mod}' mappings " \
                f"cover point indices in [0, {data_mod.num_points}]."

    def __len__(self):
        return self.data.num_nodes

    @property
    def num_points(self):
        return len(self)

    @property
    def num_node_features(self):
        return self.data.num_node_features

    def to(self, device):
        out = self.clone()
        out.data = out.data.to(device)
        out.modalities = {mod: data_mod.to(device)
                          for mod, data_mod in out.modalities.items()}
        return out

    @property
    def device(self):
        for data_mod in self.modalities.values():
            return data_mod.device

    def load(self):
        self.modalities = {mod: data_mod.load()
                           for mod, data_mod in self.modalities.items()}
        return self

    def clone(self):
        return MMData(
            self.data.clone(),
            **{mod: data_mod.clone()
             for mod, data_mod in self.modalities.items()})

    def __getitem__(self, idx):
        """
        Indexing mechanism on the points.

        Returns a new copy of the indexed MMData, with updated modality
        data and mappings. Supports torch and numpy indexing.
        """
        idx = tensor_idx(idx).to(self.device)

        # Index the Data first
        data = self.data.clone()
        for key, item in self.data:
            if torch.is_tensor(item) and item.size(0) == self.data.num_nodes:
                data[key] = data[key][idx]

        # Update the modality data and mappings wrt data key indices
        modalities = {mod: None for mod in self.modalities.keys()}
        for mod, data_mod in self.modalities.items():
            modalities[mod] = data_mod.select_points(data[self.mapping_key],
                                                     mode='pick')

        # Update point indices to the new mappings length. This is
        # important to preserve the mappings and for multimodal data
        # batching mechanisms.
        data[self.mapping_key] = torch.arange(data.num_nodes,
                                              device=self.device)

        return MMData(data, **modalities)

    def __repr__(self):
        info = [f"    data = {self.data}"]
        info = info + \
           [f"    {mod} = {data_mod}"
            for mod, data_mod in self.modalities.items()]
        info = '\n'.join(info)
        return f"{self.__class__.__name__}(\n{info}\n)"


class MMBatch(MMData):
    """
    A wrapper around MMData to create batches of multimodal data while
    leveraging the batch mechanisms for each modality attribute.

    Relies on several assumptions that MMData.debug() keeps in check. 
    """

    def __init__(self, data, **kwargs):
        super(MMBatch, self).__init__(data, **kwargs)
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
            **{mod: data_mod.clone()
             for mod, data_mod in self.modalities.items()})
        out.__sizes__ = self.__sizes__
        return out

    @staticmethod
    def from_mm_data_list(mm_data_list):
        assert isinstance(mm_data_list, list) and len(mm_data_list) > 0
        assert all([isinstance(mm_data, MMData) for mm_data in mm_data_list])
        assert all([set(mm_data.modalities.keys())
                    == set(mm_data_list[0].modalities.keys())
                    for mm_data in mm_data_list]), \
            "All MMData in the list must have the same modalities."

        # Convert list of Data to Batch
        data = Batch.from_data_list(
            [mm_data.data for mm_data in mm_data_list])

        # Convert list of modality-specific data to their batch
        # counterpart
        modalities = {mod: data_mod.get_batch_type().from_data_list(
            [mm_data.modalities[mod] for mm_data in mm_data_list])
            for mod, data_mod in mm_data_list[0].modalities.items()}

        # Instantiate the MMBatch and set the __sizes__ to allow input
        # MMData list MMBatch.recovery with to_mm_data_list()
        batch = MMBatch(data, **modalities)
        batch.__sizes__ = np.array(
            [len(mm_data) for mm_data in mm_data_list])

        return batch

    def to_mm_data_list(self):
        if self.__sizes__ is None:
            raise RuntimeError(
                'Cannot reconstruct multimodal data list from batch '
                'because the batch object was not created using '
                '`MMBatch.from_mm_data_list()`.')

        data_list = self.data.to_data_list()

        mods_dict_list = {
            mod: data_mod.to_data_list()
            for mod, data_mod in self.modalities.items()}
        mods_list_dict = [{
            mod: data_mod[i]
            for mod, data_mod in mods_dict_list.items()}
            for i in range(self.num_batch_items)]

        return [MMData(data, **modalities)
                for data, modalities
                in zip(data_list, mods_list_dict)]
