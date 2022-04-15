import torch
from torch_geometric.data import Data


class SimpleBatch(Data):
    r"""A classic batch object wrapper with
    :class:`torch_geometric.data.Data` being the base class, all its
    methods can also be used here.

    Different from the :class:`torch_geometric.data.Data`, `SimpleBatch`
    assumes all input items have the same number of points. These are
    simply stacked along a new first dimension.
    """

    def __init__(self, batch=None, **kwargs):
        super(SimpleBatch, self).__init__(**kwargs)

        self.batch = batch
        self.__data_class__ = Data

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects. 
        """
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))

        # Check if all dimensions matches and we can concatenate data
        if len(data_list) > 0:
            for i, data in enumerate(data_list[1:]):
                for key in keys:
                    if torch.is_tensor(data[key]):
                        assert data_list[0][key].shape == data[key].shape

        batch = SimpleBatch()
        batch.__data_class__ = data_list[0].__class__

        for key in keys:
            batch[key] = []

        for _, data in enumerate(data_list):
            for key in data.keys:
                item = data[key]
                batch[key].append(item)

        for key in batch.keys:
            item = batch[key][0]
            if torch.is_tensor(item):
                batch[key] = torch.stack(batch[key])
            elif isinstance(item, int) or isinstance(item, float):
                batch[key] = torch.as_tensor(batch[key])
            else:
                raise ValueError("Unsupported attribute type")

        # Set the `batch` attribute of the batch, so that we can recover
        # which point comes from which batch item
        batch.batch = torch.arange(len(data_list))

        return batch.contiguous()
        # return [batch.x.transpose(1, 2).contiguous(), batch.pos, batch.y.view(-1)]

    def to_data_list(self):
        r"""Restore the batch :class:`torch_geometric.data.Data` items
        that make up the batch.
        """
        return [
            self.__data_class__(**{k: self[k][i] for k in self.keys if k != 'batch'})
            for i in range(self.num_graphs)]

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1
