import numpy as np
from numba import njit
import torch
import copy

"""
CSR format

This format is adapted to store lists of lists of items in a tensor format. 

Rather than manipulating lists of items of variable size, we stack all items in 
a large Value tensor and keep track of the initial groupings with a Jump tensor.

The Jump tensor holds the indices where to slice the Value tensor to recover the
initial list of lists. Values[Jump[i]:Jump[i+1]] will return the
List(List(Items))[i].

NB : by convention Jumps[0] = 0, to facilitate manipulation.

When selecting items from the CSR format, special attention must be given to 
re-indexing and ordering.


Example
-------
import torch
from torch_points3d.datasets.multimodal.csr import CSRData, CSRDataBatch

n_groups = 3
n_items = 12
n_items_nested = 1000

indices = torch.sort(torch.randint(0, size=(n_items,), high=n_groups))[0]
float_values = torch.rand(n_items)
indices_nested = torch.sort(torch.randint(0, size=(n_items_nested,), high=n_items))[0]
csr_nested = CSRData(indices_nested, dense=True)

CSRData(indices, float_values, csr_nested, dense=True)
"""


class CSRData(object):
    """
    Implements the CSRData format and associated mechanisms in Torch.

    When defining a subclass A of CSRData, it is recommended to create
    an associated CSRDataBatch subclass by doing the following:
        - ABatch inherits from (A, CSRDataBatch)
        - A.get_batch_type returns ABatch
    """

    def __init__(self, jumps: torch.LongTensor, *args, dense=False, is_index_value=None):
        """
        Initialize the jumps and values.

        Values are passed as args and stored in a list. They are expected to all
        have the same size and support torch tensor indexing (i.e. they can be
        torch tensor or CSRData objects themselves).

        If `dense=True`, jumps are treated as a sorted dense tensor of
        indices to be converted into jump indices.

        Optionally, a list of booleans `is_index_value` can be passed. It must
        be the same size as *args and indicates, for each value, whether it 
        holds elements that should be treated as indices when stacking 
        CSRData objects into a CSRDataBatch. If so, the indices will be
        updated wrt the cumulative size of the batched values.
        """
        self.jumps = CSRData._sorted_indices_to_jumps(jumps) if dense else jumps
        self.values = [*args] if len(args) > 0 else None
        if is_index_value is None or is_index_value == []:
            self.is_index_value = torch.zeros(self.num_values, dtype=torch.bool)
        else:
            self.is_index_value = torch.BoolTensor(is_index_value)
        self.debug()

    def debug(self):
        assert self.num_groups >= 1, \
            "Jump indices must cover at least one group."
        assert self.jumps[0] == 0, \
            "The first jump element must always be 0."
        assert torch.all(self.jumps[1:] - self.jumps[:-1] >= 0), \
            "Jump indices must be increasing."

        if self.values is not None:
            assert isinstance(self.values, list), \
                "Values must be held in a list."
            assert all([len(v) == self.num_items for v in self.values]), \
                "All value objects must have the same size."
            assert len(self.values[0]) == self.num_items, \
                "Jumps must cover the entire range of values."
            for v in self.values:
                if isinstance(v, CSRData):
                    v.debug()

        if self.values is not None and self.is_index_value is not None:
            assert isinstance(self.is_index_value, torch.BoolTensor), \
                "is_index_value must be a torch.BoolTensor."
            assert self.is_index_value.dtype == torch.bool, \
                "is_index_value must be an tensor of booleans."
            assert self.is_index_value.ndim == 1, \
                "is_index_value must be a 1D tensor."
            assert self.is_index_value.shape[0] == self.num_values, \
                "is_index_value size must match the number of value tensors."

    def to(self, device):
        """Move the CSRData to the specified device."""
        self.jumps = self.jumps.to(device)
        for i in range(self.num_values):
            self.values[i] = self.values[i].to(device)
        return self

    def cpu(self):
        """Move the CSRData to the CPU."""
        return self.to('cpu')

    def cuda(self):
        """Move the CSRData to the first available GPU."""
        return self.to('cuda')

    @property
    def device(self):
        return self.jumps.device

    @property
    def num_groups(self):
        return self.jumps.shape[0] - 1

    @property
    def num_values(self):
        return len(self.values) if self.values is not None else 0

    @property
    def num_items(self):
        return self.jumps[-1].item()

    @staticmethod
    def get_batch_type():
        """Required by CSRDataBatch.from_csr_list."""
        return CSRDataBatch

    def clone(self):
        return copy.deepcopy(self)

    @staticmethod
    def _sorted_indices_to_jumps(indices):
        """
        Convert pre-sorted dense indices to CSR format.
        """
        assert len(indices.shape) == 1, "Only 1D indices are accepted."
        assert indices.shape[0] >= 1, "At least one group index is required."
        assert CSRData._is_sorted_numba(np.asarray(indices)), "Indices must be sorted in increasing order."
        return torch.from_numpy(CSRData._sorted_indices_to_jumps_numba(np.asarray(indices)))

    @staticmethod
    @njit(cache=True, nogil=True)
    def _sorted_indices_to_jumps_numba(indices: np.ndarray):
        # Compute the jumps
        idx_previous = indices[0]
        jumps = [0]
        for i, idx in enumerate(indices):
            if idx != idx_previous:
                jumps.append(i)
                idx_previous = idx

        # Last index must be treated separately
        jumps.append(len(indices))

        return np.asarray(jumps)

    def reindex_groups(self, group_indices: torch.LongTensor, num_groups=None):
        """
        Returns a copy of self with modified jumps to account for new groups.
        Affects the num_groups and the order of groups. Injects 0-length jumps 
        where need be. 

        By default, jumps are implicitly linked to the group indices in
        range(0, self.num_groups). 

        Here we provide new group_indices for the existing jumps, with 
        group_indices[i] corresponding to the position of existing group i in 
        the new tensor. The indices missing from group_indices account for empty
        groups to be injected.

        The num_groups specifies the number of groups in the new tensor. If not
        provided, it is inferred from the size of group_indices. 
        """
        order = torch.argsort(group_indices)

        csr_new = self[order]
        csr_new._insert_empty_groups(group_indices[order], num_groups=num_groups)
        return csr_new

    def _insert_empty_groups(self, group_indices: torch.LongTensor, num_groups=None):
        """
        Private method called when in-place reindexing groups.

        The group_indices are assumed to be sorted and group_indices[i] 
        corresponds to the position of existing group i in the new tensor. The
        indices missing from group_indices correspond to empty groups to be 
        injected.

        The num_groups specifies the number of groups in the new tensor. If not
        provided, it is inferred from the size of group_indices. 
        """
        assert self.num_groups == group_indices.shape[0], \
            "New group indices must correspond to the existing number of groups"
        assert CSRData._is_sorted_numba(np.asarray(group_indices)), \
            "New group indices must be sorted."

        if num_groups is not None:
            num_groups = max(group_indices.max() + 1, num_groups)
        else:
            num_groups = group_indices.max() + 1

        self.jumps = torch.from_numpy(CSRData._insert_empty_groups_numba(
            np.asarray(self.jumps), np.asarray(group_indices), int(num_groups)))

    @staticmethod
    @njit(cache=True, nogil=True)
    def _insert_empty_groups_numba(jumps: np.ndarray, group_indices: np.ndarray, num_groups):
        jumps_expanded = np.zeros(num_groups + 1, dtype=group_indices.dtype)
        jumps_expanded[group_indices + 1] = jumps[1:]
        jump_previous = 0
        for i in range(jumps_expanded.shape[0]):
            if jumps_expanded[i] < jump_previous:
                jumps_expanded[i] = jump_previous
            jump_previous = jumps_expanded[i]
        return jumps_expanded

    @staticmethod
    @njit(cache=True, nogil=True)
    def _is_sorted_numba(a: np.ndarray):
        for i in range(a.size - 1):
            if a[i + 1] < a[i]:
                return False
        return True

    @staticmethod
    def _index_select_jumps(jumps: torch.LongTensor, indices: torch.LongTensor):
        """
        Index selection of jumps. 

        Returns a new jump tensor with updated jumps, along with an indices tensor to
        be used to update any values tensor associated with the input jumps.
        """
        assert indices.max() <= jumps.shape[0] - 2
        jumps_updated, val_indices = CSRData._index_select_jumps_numba(np.asarray(jumps), np.asarray(indices))
        return torch.from_numpy(jumps_updated), torch.from_numpy(np.concatenate(val_indices))

    @staticmethod
    @njit(cache=True, nogil=True)
    def _index_select_jumps_numba(jumps: np.ndarray, indices: np.ndarray):
        jumps_selection = np.zeros(indices.shape[0] + 1, dtype=jumps.dtype)
        jumps_selection[1:] = np.cumsum(jumps[indices + 1] - jumps[indices])
        val_indices = [np.arange(jumps[i], jumps[i + 1]) for i in indices]
        # Can't np.concatenate the nb.list here for some reason, so we need to
        # np.concatenate outside of the @njit scope
        return jumps_selection, val_indices

    def __getitem__(self, idx):
        """
        Indexing CSRData format. Supports Numpy and torch indexing
        mechanisms.

        Return a copy of self with updated jumps and values.
        """
        if isinstance(idx, int):
            idx = torch.LongTensor([idx])
        elif isinstance(idx, list):
            idx = torch.LongTensor(idx)
        elif isinstance(idx, slice):
            idx = torch.arange(self.num_groups)[idx]
        elif isinstance(idx, np.ndarray):
            idx = torch.from_numpy(idx)
        # elif not isinstance(idx, torch.LongTensor):
        #     raise NotImplementedError
        assert idx.dtype is torch.int64, \
            "CSRData only supports int and torch.LongTensor indexing."
        assert idx.shape[0] > 0, \
            "CSRData only supports non-empty indexing. At least one " \
            "index must be provided."
        idx = idx.to(self.device)

        # Select the jumps and prepare the values indexing
        jumps, val_idx = CSRData._index_select_jumps(self.jumps, idx)

        # Copy self and edit jumps and values
        # This preserves the class for CSRData subclasses
        out = copy.deepcopy(self)
        out.jumps = jumps
        out.values = [v[val_idx] for v in self.values]
        out.debug()

        return out

    def __len__(self):
        return self.num_groups

    def __repr__(self):
        info = [f"{key}={getattr(self, key)}" for key in ['num_groups', 'num_items']]
        return f"{self.__class__.__name__}({', '.join(info)})"


class CSRDataBatch(CSRData):
    """
    Wrapper class of CSRData to build a batch from a list of CSRData
    data and reconstruct it afterwards.

    When defining a subclass A of CSRData, it is recommended to create
    an associated CSRDataBatch subclass by doing the following:
        - ABatch inherits from (A, CSRDataBatch)
        - A.get_batch_type returns ABatch
    """

    def __init__(self, jumps, *args, dense=False, is_index_value=None):
        """
        Basic constructor for a CSRDataBatch. Batches are rather
        intended to be built using the from_csr_list() method.
        """
        super(CSRDataBatch, self).__init__(
            jumps, *args, dense=dense, is_index_value=is_index_value)
        self.__sizes__ = None
        self.__csr_type__ = CSRData

    @property
    def batch_jumps(self):
        return torch.cumsum(torch.cat((torch.LongTensor([0]), self.__sizes__)), dim=0) \
            if self.__sizes__ is not None else None

    @property
    def batch_items_sizes(self):
        return self.__sizes__ if self.__sizes__ is not None else None

    @property
    def num_batch_items(self):
        return len(self.__sizes__) if self.__sizes__ is not None else 0

    def to(self, device):
        """Move the CSRDataBatch to the specified device."""
        self = super(CSRDataBatch, self).to(device)
        self.__sizes__ = self.__sizes__.to(device) \
            if self.__sizes__ is not None else None
        return self

    @staticmethod
    def from_csr_list(csr_list):
        assert isinstance(csr_list, list) and len(csr_list) > 0
        assert isinstance(csr_list[0], CSRData), \
            "All provided items must be CSRData objects."
        csr_type = type(csr_list[0])
        assert all([isinstance(csr, csr_type) for csr in csr_list]), \
            "All provided items must have the same class."
        for csr in csr_list:
            csr.debug()

        num_values = csr_list[0].num_values
        assert all([csr.num_values == num_values for csr in csr_list]), \
            "All provided items must have the same number of values."

        is_index_value = csr_list[0].is_index_value
        if is_index_value is not None:
            assert all([np.array_equal(csr.is_index_value, is_index_value) for csr in csr_list]), \
                "All provided items must have the same is_index_value."
        else:
            assert all([csr.is_index_value is None for csr in csr_list]), \
                "All provided items must have the same is_index_value."

        # Offsets are used to stack jump indices and values identified as
        # "index" value by `is_index_value` without losing the indexing
        # information they carry.
        offsets = torch.cumsum(torch.LongTensor([0] + [csr.num_items for csr in csr_list[:-1]]), dim=0)

        # Stack jumps
        jumps = torch.cat((
            torch.LongTensor([0]),
            *[csr.jumps[1:] + offset for csr, offset in zip(csr_list, offsets)],
        ), dim=0)

        # Stack values
        values = []
        for i in range(num_values):
            val_list = [csr.values[i] for csr in csr_list]
            if isinstance(csr_list[0].values[i], CSRData):
                val = CSRDataBatch.from_csr_list(val_list)
            elif is_index_value[i]:
                # "Index" values are stacked with updated indices.
                # For mappings, this implies all elements designed by the
                # index_values must be used in. There can be no element outside
                # of the range of index_values  
                idx_offsets = torch.cumsum(torch.LongTensor([0] + [v.max() + 1 for v in val_list[:-1]]), dim=0)
                val = torch.cat([v + o for v, o in zip(val_list, idx_offsets)], dim=0)
            else:
                val = torch.cat(val_list, dim=0)
            values.append(val)

        # Create the Batch object, depending on the data type
        # Default of CSRData is CSRDataBatch, but subclasses of CSRData
        # may define their own batch class inheriting from CSRDataBatch.
        batch_type = csr_type.get_batch_type()
        batch = batch_type(jumps, *values, dense=False, is_index_value=is_index_value)
        batch.__sizes__ = torch.LongTensor([csr.num_groups for csr in csr_list])
        batch.__csr_type__ = csr_type

        return batch

    def to_csr_list(self):
        if self.__sizes__ is None:
            raise RuntimeError(('Cannot reconstruct CSRData data list from batch because the ',
                                'batch object was not created using `CSRDataBatch.from_csr_list()`.'))

        group_jumps = self.batch_jumps
        item_jumps = self.jumps[group_jumps]

        # Recover jumps and index offsets for each CSRData item
        jumps = [self.jumps[group_jumps[i]:group_jumps[i + 1] + 1] - item_jumps[i]
                 for i in range(self.num_batch_items)]

        # Recover the values for each CSRData item
        values = []
        for i in range(self.num_values):
            batch_value = self.values[i]
            if isinstance(batch_value, CSRData):
                val = batch_value.to_csr_list()
            elif self.is_index_value[i]:
                val = [batch_value[item_jumps[j]:item_jumps[j + 1]]
                       - (batch_value[:item_jumps[j]].max() + 1 if j > 0 else 0)
                       for j in range(self.num_batch_items)]
            else:
                val = [batch_value[item_jumps[j]:item_jumps[j + 1]]
                       for j in range(self.num_batch_items)]
            values.append(val)
        values = [list(x) for x in zip(*values)]

        csr_list = [self.__csr_type__(j, *v, dense=False, is_index_value=self.is_index_value)
                    for j, v in zip(jumps, values)]

        return csr_list

    # def __getitem__(self, idx):
    #     """
    #     Indexing CSRDataBatch format. Supports Numpy and torch indexing
    #     mechanisms.
    #
    #     Only allows for batch-contiguous indexing as other indexes would
    #     break the batches. This means indices linking to the same batch
    #     are contiguous and preserve the original batch order.
    #
    #     Return a copy of self with updated batches, jumps and values.
    #     """
    #     if isinstance(idx, int):
    #         idx = torch.LongTensor([idx])
    #     elif isinstance(idx, list):
    #         idx = torch.LongTensor(idx)
    #     elif isinstance(idx, slice):
    #         idx = torch.arange(self.num_groups)[idx]
    #     elif isinstance(idx, np.ndarray):
    #         idx = torch.from_numpy(idx)
    #     assert idx.dtype is torch.int64, \
    #         "CSRData only supports int and torch.LongTensor indexing"
    #
    #     # Find the batch each index falls into and ensure indices are
    #     # batch-contiguous. Otherwise indexing the CSRDataBatch would
    #     # break the batching.
    #     idx_batch_ids = torch.bucketize(idx, self.batch_jumps[1:], right=True)
    #
    #     # Recover the indexing to be separately applied to each CSRData
    #     # item in the CSRDataBatch. If the index is not sorted in a
    #     # batch-contiguous fashion, this will raise an error.
    #     idx_batch_jumps = CSRData._sorted_indices_to_jumps(idx_batch_ids)
    #     idx_list = [
    #         (
    #             idx_batch_ids[idx_batch_jumps[i]],
    #             idx[idx_batch_jumps[i]:idx_batch_jumps[i+1]] - self.batch_jumps[idx_batch_ids[idx_batch_jumps[i]]]
    #         )
    #         for i in range(len(idx_batch_jumps) - 1)
    #     ]
    #
    #     # Convert the CSRDataBatch to its list of CSRData and index the
    #     # proper CSRData objects with the associated indices.
    #     # REMARK: some CSRData items may be discarded in the process,
    #     # if not all batch items are represented in the input idx.
    #     csr_list = self.to_csr_list()
    #     csr_list = [
    #         csr_list[i_csr][idx_csr] for i_csr, idx_csr in idx_list
    #     ]
    #
    #     return CSRDataBatch.from_csr_list(csr_list)

    def __getitem__(self, idx):
        """
        Indexing CSRDataBatch format. Supports Numpy and torch indexing
        mechanisms.

        Indexing a CSRDataBatch breaks the reversible batching
        mechanism between `from_csr_list` and `to_csr_list`. As a
        result, the indexed output is a __csr_type__ from which the
        original items can no longer be retrieved with to_csr_list`.
        """
        csr_batch = super(CSRDataBatch, self).__getitem__(idx)
        out = self.__csr_type__(
            csr_batch.jumps, *csr_batch.values, dense=False,
            is_index_value=csr_batch.is_index_value)
        out.debug()
        return out

    def __repr__(self):
        info = [f"{key}={getattr(self, key)}"
                for key in ['num_batch_items', 'num_groups', 'num_items']]
        return f"{self.__class__.__name__}({', '.join(info)})"
