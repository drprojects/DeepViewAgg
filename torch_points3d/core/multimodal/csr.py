import numpy as np
# from numba import njit
import torch
import copy
from torch_points3d.utils.multimodal import tensor_idx

"""
CSR format

This format is adapted to store lists of lists of items in a tensor format. 

Rather than manipulating lists of items of variable size, we stack all items in 
a large Value tensor and keep track of the initial groupings with a pointer 
tensor.

The pointer tensor holds the indices where to slice the Value tensor to recover 
the initial list of lists. Values[pointer[i]:pointer[i+1]] will return the
List(List(Items))[i].

NB : by convention pointers[0] = 0, to facilitate manipulation.

When selecting items from the CSR format, special attention must be given to 
re-indexing and ordering.


Example
-------
import torch
from torch_points3d.core.multimodal.csr import CSRData, CSRBatch

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
    an associated CSRBatch subclass by doing the following:
        - ABatch inherits from (A, CSRBatch)
        - A.get_batch_type() returns ABatch
    """

    def __init__(self, pointers: torch.LongTensor, *args, dense=False,
                 is_index_value=None):
        """
        Initialize the pointers and values.

        Values are passed as args and stored in a list. They are expected to all
        have the same size and support torch tensor indexing (i.e. they can be
        torch tensor or CSRData objects themselves).

        If `dense=True`, pointers are treated as a sorted dense tensor of
        indices to be converted into pointer indices.

        Optionally, a list of booleans `is_index_value` can be passed. It must
        be the same size as *args and indicates, for each value, whether it 
        holds elements that should be treated as indices when stacking 
        CSRData objects into a CSRBatch. If so, the indices will be
        updated wrt the cumulative size of the batched values.
        """
        self.pointers = CSRData._sorted_indices_to_pointers(pointers) \
            if dense else pointers
        self.values = [*args] if len(args) > 0 else None
        if is_index_value is None or is_index_value == []:
            self.is_index_value = torch.zeros(self.num_values, dtype=torch.bool)
        else:
            self.is_index_value = torch.BoolTensor(is_index_value)
        self.debug()

    def debug(self):
        assert self.num_groups >= 1, \
            "pointer indices must cover at least one group."
        assert self.pointers[0] == 0, \
            "The first pointer element must always be 0."
        assert torch.all(self.pointers[1:] - self.pointers[:-1] >= 0), \
            "pointer indices must be increasing."

        if self.values is not None:
            assert isinstance(self.values, list), \
                "Values must be held in a list."
            assert all([len(v) == self.num_items for v in self.values]), \
                "All value objects must have the same size."
            assert len(self.values[0]) == self.num_items, \
                "pointers must cover the entire range of values."
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
        out = self.clone()
        out.pointers = out.pointers.to(device)
        for i in range(out.num_values):
            out.values[i] = out.values[i].to(device)
        return out

    def cpu(self):
        """Move the CSRData to the CPU."""
        return self.to('cpu')

    def cuda(self):
        """Move the CSRData to the first available GPU."""
        return self.to('cuda')

    @property
    def device(self):
        return self.pointers.device

    @property
    def num_groups(self):
        return self.pointers.shape[0] - 1

    @property
    def num_values(self):
        return len(self.values) if self.values is not None else 0

    @property
    def num_items(self):
        return self.pointers[-1].item()

    @staticmethod
    def get_batch_type():
        """Required by CSRBatch.from_csr_list."""
        return CSRBatch

    def clone(self):
        """
        Shallow copy of self. This may cause issues for certain types of
        downstream operations but it saves time and memory. In practice,
        it shouldn't in this project.
        """
        out = copy.copy(self)
        out.pointers = copy.copy(self.pointers)
        out.values = copy.copy(self.values)
        return out

    @staticmethod
    def _sorted_indices_to_pointers(indices: torch.LongTensor):
        """
        Convert pre-sorted dense indices to CSR format.
        """
        device = indices.device
        assert len(indices.shape) == 1, "Only 1D indices are accepted."
        assert indices.shape[0] >= 1, "At least one group index is required."
        assert CSRData._is_sorted(indices), \
            "Indices must be sorted in increasing order."
        pointers = torch.cat([
            torch.LongTensor([0]).to(device),
            torch.where(indices[1:] > indices[:-1])[0] + 1,
            torch.LongTensor([indices.shape[0]]).to(device)])
        return pointers

    # @staticmethod
    # def _sorted_indices_to_pointers(indices):
    #     """
    #     Convert pre-sorted dense indices to CSR format.
    #     """
    #     device = indices.device
    #     assert len(indices.shape) == 1, "Only 1D indices are accepted."
    #     assert indices.shape[0] >= 1, "At least one group index is required."
    #     assert CSRData._is_sorted_numba(np.asarray(indices.cpu())), \
    #         "Indices must be sorted in increasing order."
    #     sorted_indices = CSRData._sorted_indices_to_pointers_numba(np.asarray(
    #         indices.cpu()))
    #     return torch.from_numpy(sorted_indices).to(device)

    # @staticmethod
    # @njit(cache=True, nogil=True)
    # def _sorted_indices_to_pointers_numba(indices: np.ndarray):
    #     # Compute the pointers
    #     idx_previous = indices[0]
    #     pointers = [0]
    #     for i, idx in enumerate(indices):
    #         if idx != idx_previous:
    #             pointers.append(i)
    #             idx_previous = idx
    #
    #     # Last index must be treated separately
    #     pointers.append(len(indices))
    #
    #     return np.asarray(pointers)

    def reindex_groups(self, group_indices: torch.LongTensor, num_groups=None):
        """
        Returns a copy of self with modified pointers to account for new groups.
        Affects the num_groups and the order of groups. Injects 0-length
        pointers where need be.

        By default, pointers are implicitly linked to the group indices in
        range(0, self.num_groups). 

        Here we provide new group_indices for the existing pointers, with 
        group_indices[i] corresponding to the position of existing group i in 
        the new tensor. The indices missing from group_indices account for empty
        groups to be injected.

        The num_groups specifies the number of groups in the new tensor. If not
        provided, it is inferred from the size of group_indices. 
        """
        order = torch.argsort(group_indices)
        csr_new = self[order].insert_empty_groups(
            group_indices[order],
            num_groups=num_groups)
        return csr_new

    def insert_empty_groups(self, group_indices: torch.LongTensor,
                            num_groups=None):
        """
        Method called when in-place reindexing groups.

        The group_indices are assumed to be sorted and group_indices[i]
        corresponds to the position of existing group i in the new tensor. The
        indices missing from group_indices correspond to empty groups to be
        injected.

        The num_groups specifies the number of groups in the new tensor. If not
        provided, it is inferred from the size of group_indices.
        """
        assert self.num_groups == group_indices.shape[0], \
            "New group indices must correspond to the existing number of groups"
        assert CSRData._is_sorted(group_indices), \
            "New group indices must be sorted."

        if num_groups is not None:
            num_groups = max(group_indices.max() + 1, num_groups)
        else:
            num_groups = group_indices.max() + 1

        starts = torch.cat([
            torch.LongTensor([-1]).to(self.device),
            group_indices.to(self.device)])
        ends = torch.cat([
            group_indices.to(self.device),
            torch.LongTensor([num_groups]).to(self.device)])
        repeats = ends - starts
        self.pointers = torch.repeat_interleave(self.pointers, repeats)

        return self

    # def insert_empty_groups(self, group_indices: torch.LongTensor,
    #                         num_groups=None):
    #     """
    #     Method called when in-place reindexing groups.
    #
    #     The group_indices are assumed to be sorted and group_indices[i]
    #     corresponds to the position of existing group i in the new tensor. The
    #     indices missing from group_indices correspond to empty groups to be
    #     injected.
    #
    #     The num_groups specifies the number of groups in the new tensor. If not
    #     provided, it is inferred from the size of group_indices.
    #     """
    #     assert self.num_groups == group_indices.shape[0], \
    #         "New group indices must correspond to the existing number of groups"
    #     assert CSRData._is_sorted_numba(np.asarray(group_indices.cpu())), \
    #         "New group indices must be sorted."
    #
    #     if num_groups is not None:
    #         num_groups = max(group_indices.max() + 1, num_groups)
    #     else:
    #         num_groups = group_indices.max() + 1
    #
    #     self.pointers = torch.from_numpy(CSRData._insert_empty_groups_numba(
    #         np.asarray(self.pointers.cpu()), np.asarray(group_indices.cpu()),
    #         int(num_groups))).to(self.device)
    #
    #     return self

    # @staticmethod
    # @njit(cache=True, nogil=True)
    # def _insert_empty_groups_numba(pointers: np.ndarray,
    #                                group_indices: np.ndarray, num_groups):
    #     pointers_expanded = np.zeros(num_groups + 1, dtype=group_indices.dtype)
    #     pointers_expanded[group_indices + 1] = pointers[1:]
    #     pointer_previous = 0
    #     for i in range(pointers_expanded.shape[0]):
    #         if pointers_expanded[i] < pointer_previous:
    #             pointers_expanded[i] = pointer_previous
    #         pointer_previous = pointers_expanded[i]
    #     return pointers_expanded

    # @staticmethod
    # @njit(cache=True, nogil=True)
    # def _is_sorted_numba(a: np.ndarray):
    #     for i in range(a.size - 1):
    #         if a[i + 1] < a[i]:
    #             return False
    #     return True

    @staticmethod
    def _is_sorted(a: torch.Tensor):
        return torch.all(a[:-1] <= a[1:])

    @staticmethod
    def _index_select_pointers(pointers: torch.LongTensor,
                               indices: torch.LongTensor):
        """
        Index selection of pointers.

        Returns a new pointer tensor with updated pointers, along with an
        indices tensor to be used to update any values tensor associated with
        the input pointers.
        """
        assert indices.max() <= pointers.shape[0] - 2
        device = pointers.device

        # Create the new pointers
        pointers_new = torch.cat([
            torch.zeros(1, dtype=pointers.dtype, device=device),
            torch.cumsum(pointers[indices + 1] - pointers[indices], 0)])

        # Create the indexing tensor to select and order values.
        # Simply, we could have used a list of slices but we want to
        # avoid for loops and list concatenations to benefit from torch
        # capabilities.
        sizes = pointers_new[1:] - pointers_new[:-1]
        val_idx = torch.arange(pointers_new[-1]).to(device)
        val_idx -= torch.repeat_interleave(
            torch.arange(pointers_new[-1] + 1)[pointers_new[:-1]],
            sizes).to(device)
        val_idx += torch.repeat_interleave(pointers[indices], sizes).to(device)

        return pointers_new, val_idx

    # @staticmethod
    # def _index_select_pointers(pointers: torch.LongTensor,
    #                            indices: torch.LongTensor):
    #     """
    #     Index selection of pointers.
    #
    #     Returns a new pointer tensor with updated pointers, along with an
    #     indices tensor to be used to update any values tensor associated with
    #     the input pointers.
    #     """
    #     assert indices.max() <= pointers.shape[0] - 2
    #     device = pointers.device
    #     pointers_updated, val_indices = CSRData._index_select_pointers_numba(
    #         np.asarray(pointers.cpu()), np.asarray(indices.cpu()))
    #     return torch.from_numpy(pointers_updated).to(device), torch.from_numpy(
    #         np.concatenate(val_indices)).to(device)
    #
    # @staticmethod
    # @njit(cache=True, nogil=True)
    # def _index_select_pointers_numba(pointers: np.ndarray,
    #                                  indices: np.ndarray):
    #     pointers_selection = np.zeros(indices.shape[0] + 1,
    #                                   dtype=pointers.dtype)
    #     pointers_selection[1:] = np.cumsum(
    #         pointers[indices + 1] - pointers[indices])
    #     val_indices = [np.arange(pointers[i], pointers[i + 1])
    #                    for i in indices]
    #     # Can't np.concatenate the nb.list here for some reason, so we
    #     # need to np.concatenate outside of the @njit scope
    #     return pointers_selection, val_indices

    def __getitem__(self, idx):
        """
        Indexing CSRData format. Supports Numpy and torch indexing
        mechanisms.

        Return a copy of self with updated pointers and values.
        """
        idx = tensor_idx(idx).to(self.device)

        # Select the pointers and prepare the values indexing
        pointers, val_idx = CSRData._index_select_pointers(self.pointers, idx)

        # Shallow copy self and edit pointers and values. This
        # preserves the class for CSRData subclasses.
        out = self.clone()
        out.pointers = pointers
        out.values = [v[val_idx] for v in self.values]
        out.debug()

        return out

    def __len__(self):
        return self.num_groups

    def __repr__(self):
        info = [f"{key}={getattr(self, key)}"
                for key in ['num_groups', 'num_items', 'device']]
        return f"{self.__class__.__name__}({', '.join(info)})"


class CSRBatch(CSRData):
    """
    Wrapper class of CSRData to build a batch from a list of CSRData
    data and reconstruct it afterwards.

    When defining a subclass A of CSRData, it is recommended to create
    an associated CSRBatch subclass by doing the following:
        - ABatch inherits from (A, CSRBatch)
        - A.get_batch_type() returns ABatch
    """

    def __init__(self, pointers, *args, dense=False, is_index_value=None):
        """
        Basic constructor for a CSRBatch. Batches are rather
        intended to be built using the from_csr_list() method.
        """
        super(CSRBatch, self).__init__(
            pointers, *args, dense=dense, is_index_value=is_index_value)
        self.__sizes__ = None
        self.__csr_type__ = CSRData

    @property
    def batch_pointers(self):
        return torch.cumsum(
            torch.cat((torch.LongTensor([0]), self.__sizes__)), dim=0) \
            if self.__sizes__ is not None else None

    @property
    def batch_items_sizes(self):
        return self.__sizes__ if self.__sizes__ is not None else None

    @property
    def num_batch_items(self):
        return len(self.__sizes__) if self.__sizes__ is not None else 0

    def to(self, device):
        """Move the CSRBatch to the specified device."""
        out = super(CSRBatch, self).to(device)
        out.__sizes__ = self.__sizes__.to(device) \
            if self.__sizes__ is not None else None
        return out

    @staticmethod
    def from_csr_list(csr_list):
        assert isinstance(csr_list, list) and len(csr_list) > 0
        assert isinstance(csr_list[0], CSRData), \
            "All provided items must be CSRData objects."
        csr_type = type(csr_list[0])
        assert all([isinstance(csr, csr_type) for csr in csr_list]), \
            "All provided items must have the same class."
        device = csr_list[0].device
        assert all([csr.device == device for csr in csr_list]), \
            "All provided items must be on the same device."
        num_values = csr_list[0].num_values
        assert all([csr.num_values == num_values for csr in csr_list]), \
            "All provided items must have the same number of values."
        is_index_value = csr_list[0].is_index_value
        if is_index_value is not None:
            assert all([np.array_equal(csr.is_index_value, is_index_value)
                        for csr in csr_list]), \
                "All provided items must have the same is_index_value."
        else:
            assert all([csr.is_index_value is None for csr in csr_list]), \
                "All provided items must have the same is_index_value."
        for csr in csr_list:
            csr.debug()

        # Offsets are used to stack pointer indices and values identified as
        # "index" value by `is_index_value` without losing the indexing
        # information they carry.
        offsets = torch.cumsum(torch.LongTensor(
            [0] + [csr.num_items for csr in csr_list[:-1]]), dim=0).to(device)

        # Stack pointers
        pointers = torch.cat((
            torch.LongTensor([0]).to(device),
            *[csr.pointers[1:] + offset
              for csr, offset in zip(csr_list, offsets)],
        ), dim=0)

        # Stack values
        values = []
        for i in range(num_values):
            val_list = [csr.values[i] for csr in csr_list]
            if isinstance(csr_list[0].values[i], CSRData):
                val = CSRBatch.from_csr_list(val_list)
            elif is_index_value[i]:
                # "Index" values are stacked with updated indices.
                # For mappings, this implies all elements designed by the
                # index_values must be used in. There can be no element outside
                # of the range of index_values  
                idx_offsets = torch.cumsum(torch.LongTensor(
                    [0] + [v.max() + 1 for v in val_list[:-1]]),
                    dim=0).to(device)
                val = torch.cat([v + o for v, o in zip(val_list, idx_offsets)],
                    dim=0)
            else:
                val = torch.cat(val_list, dim=0)
            values.append(val)

        # Create the Batch object, depending on the data type
        # Default of CSRData is CSRBatch, but subclasses of CSRData
        # may define their own batch class inheriting from CSRBatch.
        batch_type = csr_type.get_batch_type()
        batch = batch_type(pointers, *values, dense=False,
                           is_index_value=is_index_value)
        batch.__sizes__ = torch.LongTensor([csr.num_groups for csr in csr_list])
        batch.__csr_type__ = csr_type

        return batch

    def to_csr_list(self):
        if self.__sizes__ is None:
            raise RuntimeError(
                'Cannot reconstruct CSRData data list from batch because the '
                'batch object was not created using '
                '`CSRBatch.from_csr_list()`.')

        group_pointers = self.batch_pointers
        item_pointers = self.pointers[group_pointers]

        # Recover pointers and index offsets for each CSRData item
        pointers = [
            self.pointers[group_pointers[i]:group_pointers[i + 1] + 1]
            - item_pointers[i]
            for i in range(self.num_batch_items)]

        # Recover the values for each CSRData item
        values = []
        for i in range(self.num_values):
            batch_value = self.values[i]
            if isinstance(batch_value, CSRData):
                val = batch_value.to_csr_list()
            elif self.is_index_value[i]:
                val = [
                    batch_value[item_pointers[j]:item_pointers[j + 1]]
                    - (batch_value[:item_pointers[j]].max() + 1 if j > 0 else 0)
                    for j in range(self.num_batch_items)]
            else:
                val = [batch_value[item_pointers[j]:item_pointers[j + 1]]
                       for j in range(self.num_batch_items)]
            values.append(val)
        values = [list(x) for x in zip(*values)]

        csr_list = [
            self.__csr_type__(
                j, *v, dense=False, is_index_value=self.is_index_value)
            for j, v in zip(pointers, values)]

        return csr_list

    # def __getitem__(self, idx):
    #     """
    #     Indexing CSRBatch format. Supports Numpy and torch indexing
    #     mechanisms.
    #
    #     Only allows for batch-contiguous indexing as other indexes would
    #     break the batches. This means indices linking to the same batch
    #     are contiguous and preserve the original batch order.
    #
    #     Return a copy of self with updated batches, pointers and values.
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
    #     # batch-contiguous. Otherwise indexing the CSRBatch would
    #     # break the batching.
    #     idx_batch_ids = torch.bucketize(idx, self.batch_pointers[1:], right=True)
    #
    #     # Recover the indexing to be separately applied to each CSRData
    #     # item in the CSRBatch. If the index is not sorted in a
    #     # batch-contiguous fashion, this will raise an error.
    #     idx_batch_pointers = CSRData._sorted_indices_to_pointers(idx_batch_ids)
    #     idx_list = [
    #         (
    #             idx_batch_ids[idx_batch_pointers[i]],
    #             idx[idx_batch_pointers[i]:idx_batch_pointers[i+1]] - self.batch_pointers[idx_batch_ids[idx_batch_pointers[i]]]
    #         )
    #         for i in range(len(idx_batch_pointers) - 1)
    #     ]
    #
    #     # Convert the CSRBatch to its list of CSRData and index the
    #     # proper CSRData objects with the associated indices.
    #     # REMARK: some CSRData items may be discarded in the process,
    #     # if not all batch items are represented in the input idx.
    #     csr_list = self.to_csr_list()
    #     csr_list = [
    #         csr_list[i_csr][idx_csr] for i_csr, idx_csr in idx_list
    #     ]
    #
    #     return CSRBatch.from_csr_list(csr_list)

    def __getitem__(self, idx):
        """
        Indexing CSRBatch format. Supports Numpy and torch indexing
        mechanisms.

        Indexing a CSRBatch breaks the reversible batching
        mechanism between `from_csr_list` and `to_csr_list`. As a
        result, the indexed output is a __csr_type__ from which the
        original items can no longer be retrieved with to_csr_list`.
        """
        csr_batch = super(CSRBatch, self).__getitem__(idx)
        out = self.__csr_type__(
            csr_batch.pointers, *csr_batch.values, dense=False,
            is_index_value=csr_batch.is_index_value)
        out.debug()
        return out

    def __repr__(self):
        info = [f"{key}={getattr(self, key)}"
                for key in ['num_batch_items', 'num_groups', 'num_items',
                            'device']]
        return f"{self.__class__.__name__}({', '.join(info)})"
