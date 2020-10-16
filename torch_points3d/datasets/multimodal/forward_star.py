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

class ForwardStar(object):
    """
    Implements the ForwardStar format and associated mechanisms in Numpy.
    """
    
    def __init__(self, jumps, *args, dense=False):
        """
        Initialize the jumps and values. Values are passed as args and stored 
        in a list. If dense=True, jumps are treated as a dense array of indices
        to be converted into jump indices.
        """
        self.jumps = ForwardStar.indices_to_jumps(jumps) if dense else jumps
        self.values = [*args] if args else None


    @property
    def num_groups(self):
        return self.jumps.shape[0] - 1  


    @staticmethod    
    def indices_to_jumps(indices):
        """
        Convert dense format to forward star. Indices are assumed to be already
        sorted, if sorting is necessary.  
        """
        assert len(indices.shape) == 1
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
        jumps.append(i+1)

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

        if num_groups:
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
        for i in range(a.size-1):
             if a[i+1] < a[i] :
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

        if self.values:
            return ForwardStar(jumps, *[v[val_idx] for v in self.values], dense=False)
        else:
            return ForwardStar(jumps, dense=False)


    def __len__(self):
        return self.num_groups


    def __repr__(self): 
        return self.__class__.__name__


