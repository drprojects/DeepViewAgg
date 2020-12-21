import numpy as np
from numba import njit

"""
Forward Star format

This format is adapted to store lists of lists of items in an array format. 

Rather than manipulating lists of items of variable size, we stack all items in 
a large Value array and keep track of the initial groupings with a Jump array.

The Jump array holds the indices where to slice the Value array to recover the
initial list of lists. Values[Jump[i]:Jump[i+1]] will return the
List(List(Items))[i].

NB : by convention Jumps[0] = 0, to facilitate manipulation.

When electing items from the Forward Star format, special attention must be 
given to re-indexing and ordering.
"""

# TODO : better naming convention, rather than CSR
# TODO : modality-specific mappings (pixels, features, batching, resolution updates)
# TODO : mappings must also carry projection features
# TODO : mapping updates: main resolution resampling from idx, modality resampling from res ratio.
#  Expand to modality-specific

class ForwardStar(object):
    """
    Implements the ForwardStar format and associated mechanisms in Numpy.
    """

    def __init__(self, jumps, *args, dense=False, is_index_value=None):
        """
        Initialize the jumps and values.

        Values are passed as args and stored in a list. They are expected to all
        have the same size and support numpy array indexing (i.e. they can be 
        numpy arrays or ForwardStar objects themselves).

        If dense=True, jumps are treated as a dense array of indices to be 
        converted into jump indices. 

        Optionnally, a list of booleans is_index_value can be passed. It must 
        be the same size as *args and indicates, for each value, whether it 
        holds elements that should be treated as indices when stacking 
        ForwardStar objects into a ForwardStarBatch. If so, the indices will be 
        updated wrt the cumulative size of the batched values.
        """
        self.jumps = ForwardStar.indices_to_jumps(jumps) if dense else jumps
        self.values = [*args] if len(args) > 0 else None
        if is_index_value is None:
            self.is_index_value = np.array([False] * self.num_values)
        else:
            self.is_index_value = np.asarray(is_index_value)
        self.debug()

    def debug(self):
        assert self.num_groups >= 1, \
            "Jump indices must cover at least one group."
        assert self.jumps[0] == 0, \
            "The first jump element must always be 0."
        assert np.all(self.jumps[1:] - self.jumps[:-1] >= 0), \
            "Jump indices must be increasing."

        if self.values is not None:
            assert isinstance(self.values, list), \
                "Values must be held in a list."
            assert all([len(v) == self.num_items for v in self.values]), \
                "All value objects must have the same size."
            assert len(self.values[0]) == self.num_items, \
                "Jumps must cover the entire range of values."
            for v in self.values:
                if isinstance(v, ForwardStar):
                    v.debug()

        if self.values is not None and self.is_index_value is not None:
            assert isinstance(self.is_index_value, np.ndarray), \
                "is_index_value must be a numpy array."
            assert self.is_index_value.dtype == np.bool, \
                "is_index_value must be an array of booleans."
            assert self.is_index_value.ndim == 1, \
                "is_index_value must be a 1D array."
            assert self.is_index_value.size == self.num_values, \
                "is_index_value size must match the number of value arrays."

    @property
    def num_groups(self):
        return self.jumps.shape[0] - 1

    @property
    def num_values(self):
        return len(self.values) if self.values is not None else 0

    @property
    def num_items(self):
        return self.jumps[-1]

    @staticmethod
    def indices_to_jumps(indices):
        """
        Convert dense format to forward star. Indices are assumed to be ALREADY
        SORTED, if sorting is necessary.  
        """
        assert len(indices.shape) == 1, "Only 1D indices are accepted."
        assert indices.shape[0] >= 1, "At least one group index is required."
        return ForwardStar.indices_to_jumps_numba(indices)

    @staticmethod
    @njit
    def indices_to_jumps_numba(indices):
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

    def reindex_groups(self, group_indices, num_groups=None):
        """
        Returns a copy of self with modified jumps to account for new groups.
        Affects the num_groups and the order of groups. Injects 0-length jumps 
        where need be. 

        By default, jumps are implicitly linked to the group indices in
        range(0, self.num_groups). 

        Here we provide new group_indices for the existing jumps, with 
        group_indices[i] corresponding to the position of existing group i in 
        the new array. The indices missing from group_indices account for empty 
        groups to be injected.

        The num_groups specifies the number of groups in the new array. If not
        provided, it is inferred from the size of group_indices. 
        """
        group_indices = np.asarray(group_indices)
        order = np.argsort(group_indices)

        fs_new = self[order]
        fs_new.__insert_empty_groups(group_indices[order], num_groups=num_groups)
        return fs_new

    def __insert_empty_groups(self, group_indices, num_groups=None):
        """
        Private method called when in-place reindexing groups.

        The group_indices are assumed to be sorted and group_indices[i] 
        corresponds to the position of existing group i in the new array. The 
        indices missing from group_indices correspond to empty groups to be 
        injected.

        The num_groups specifies the number of groups in the new array. If not
        provided, it is inferred from the size of group_indices. 
        """
        group_indices = np.asarray(group_indices)
        assert self.num_groups == group_indices.shape[0], ("New group indices must correspond to ",
                                                           "the existing number of groups")
        assert ForwardStar.is_sorted(group_indices), "New group indices must be sorted."

        if num_groups is not None:
            num_groups = max(group_indices.max() + 1, num_groups)
        else:
            num_groups = group_indices.max() + 1

        self.jumps = ForwardStar.insert_empty_groups_numba(self.jumps, group_indices, num_groups)

    @staticmethod
    @njit
    def insert_empty_groups_numba(jumps, group_indices, num_groups):
        jumps_expanded = np.zeros(num_groups + 1, dtype=group_indices.dtype)
        jumps_expanded[group_indices + 1] = jumps[1:]
        jump_previous = 0
        for i in range(jumps_expanded.shape[0]):
            if jumps_expanded[i] < jump_previous:
                jumps_expanded[i] = jump_previous
            jump_previous = jumps_expanded[i]
        return jumps_expanded

    @staticmethod
    @njit
    def is_sorted(a):
        for i in range(a.size - 1):
            if a[i + 1] < a[i]:
                return False
        return True

    @staticmethod
    def index_select_jumps(jumps, indices):
        """
        Index selection of jumps. 

        Returns a new jump array with updated jumps, along with an indices array to
        be used to update any values array associated with the input jumps.   
        """
        assert indices.max() <= jumps.shape[0] - 2
        jumps_updated, val_indices = ForwardStar.index_select_jumps_numba(jumps, indices)
        return jumps_updated, np.concatenate(val_indices)

    @staticmethod
    @njit
    def index_select_jumps_numba(jumps, indices):
        jumps_selection = np.zeros(indices.shape[0] + 1, dtype=jumps.dtype)
        jumps_selection[1:] = np.cumsum(jumps[indices + 1] - jumps[indices])
        val_indices = [np.arange(jumps[i], jumps[i + 1]) for i in indices]
        # Can't np.concatenate  the nb.list here for some reason, so we need to 
        # np.concatenate outside of the @njit scope
        return jumps_selection, val_indices

    def __getitem__(self, idx):
        """
        Indexing ForwardStar format. Supports Numpy indexing mechanisms.

        Return a new ForwardStar object with updated jumps and values.
        """
        if isinstance(idx, int):
            idx = np.array([idx])
        idx = np.asarray(idx)
        assert idx.dtype == np.int, "ForwardStar only supports int and numpy array indexing"

        jumps, val_idx = ForwardStar.index_select_jumps(self.jumps, idx)

        if self.values is not None:
            return ForwardStar(jumps, *[v[val_idx] for v in self.values], dense=False,
                               is_index_value=self.is_index_value)
        else:
            return ForwardStar(jumps, dense=False)

    def __len__(self):
        return self.num_groups

    def __repr__(self):
        info = [f"{key}={getattr(self, key)}" for key in ['num_groups', 'num_items']]
        return f"{self.__class__.__name__}({', '.join(info)})"


class ForwardStarBatch(ForwardStar):
    """
    Wrapper class of ForwardStar to build a batch from a list of ForwardStar 
    data and reconstruct it afterwards.  
    """

    def __init__(self, jumps, *args, dense=False, is_index_value=None):
        """
        Basic constructor for a ForwardStarBatch. Batches are rather
        intendended to be built using the from_forward_star_list() method.
        """
        super(ForwardStarBatch, self).__init__(jumps, *args, dense=dense,
                                               is_index_value=is_index_value)
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
        return len(self.__sizes__) if self.__sizes__ is not None else 0

    @staticmethod
    def from_forward_star_list(fs_list):
        assert isinstance(fs_list, list) and len(fs_list) > 0
        assert all([isinstance(fs, ForwardStar) for fs in fs_list]), \
            "All provided items must be ForwardStar objects."
        for fs in fs_list:
            fs.debug()

        num_values = fs_list[0].num_values
        assert all([fs.num_values == num_values for fs in fs_list]), \
            "All provided items must have the same number of values."

        is_index_value = fs_list[0].is_index_value
        if is_index_value is not None:
            assert all([np.array_equal(fs.is_index_value, is_index_value) for fs in fs_list]), \
                "All provided items must have the same is_index_value."
        else:
            assert all([fs.is_index_value is None for fs in fs_list]), \
                "All provided items must have the same is_index_value."

        # Offsets are used to stack jump indices and values identified as 
        # is_index_value without losing the indexing information they carry
        offsets = np.cumsum(np.concatenate(([0], [fs.num_items for fs in fs_list[:-1]])))

        # Stack jumps
        jumps = np.concatenate((
            [0],
            *[fs.jumps[1:] + offset for fs, offset in zip(fs_list, offsets)],
        )).astype(np.int)

        # Stack values
        values = []
        for i in range(num_values):
            val_list = [fs.values[i] for fs in fs_list]
            if isinstance(fs_list[0].values[i], ForwardStar):
                val = ForwardStarBatch.from_forward_star_list(val_list)
            elif is_index_value[i]:
                # Index values are stacked with updated indices.
                # For mappings, this implies all elements designed by the
                # index_values must be used in. There can be no element outside
                # of the range of index_values  
                idx_offsets = np.cumsum(np.concatenate(([0], [v.max() + 1 for v in val_list[:-1]])))
                val = np.concatenate([v + o for v, o in zip(val_list, idx_offsets)])
            else:
                val = np.concatenate(val_list)
            values.append(val)

        # Create the ForwardStarBatch
        batch = ForwardStarBatch(jumps, *values, dense=False, is_index_value=is_index_value)
        batch.__sizes__ = np.array([fs.num_groups for fs in fs_list])

        return batch

    def to_forward_star_list(self):
        if self.__sizes__ is None:
            raise RuntimeError(('Cannot reconstruct ForwardStar data list from batch because the ',
                                'batch object was not created using `ForwardStarBatch.from_forward_star_list()`.'))

        group_jumps = self.batch_jumps
        item_jumps = self.jumps[group_jumps]

        # Recover jumps and index offsets
        jumps = [self.jumps[group_jumps[i]:group_jumps[i + 1] + 1] - item_jumps[i]
                 for i in range(self.num_batch_items)]

        values = []
        for i in range(self.num_values):
            batch_value = self.values[i]
            if isinstance(batch_value, ForwardStar):
                val = batch_value.to_forward_star_list()
            elif self.is_index_value[i]:
                val = [batch_value[item_jumps[j]:item_jumps[j + 1]]
                       - (batch_value[:item_jumps[j]].max() + 1 if j > 0 else 0)
                       for j in range(self.num_batch_items)]
            else:
                val = [batch_value[item_jumps[j]:item_jumps[j + 1]]
                       for j in range(self.num_batch_items)]
            values.append(val)
        values = [list(x) for x in zip(*values)]

        fs_list = [ForwardStar(j, *v, dense=False, is_index_value=self.is_index_value)
                   for j, v in zip(jumps, values)]

        return fs_list
