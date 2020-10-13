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


    def __repr__(self): 
        return self.__class__.__name__


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



    def inject_empty_groups(self, indices, idx_max=None):


        Hey, gotta change this to a more general-purpose, name-explicit group_reindexing mechanism!
        

        """
        By default, jumps are implicitly linked to the group indices
        range(0, jumps.shape[0])

        Update the jumps for the complete array of indices range(0, idx_max).

        Indices are assumed to be sorted and to correspond to the provided jumps.
        The indices with no jump will have 0-size jumps in the returned jumps. 
        """
        assert self.jumps.shape[0] == indices.shape[0] + 1
        assert is_sorted(indices), "New jump indices must be sorted."

        idx_max = max(indices.max(), idx_max) if idx_max else indices.max()

        jumps_expanded = np.zeros(idx_max + 2, dtype=indices.dtype)
        jumps_expanded[indices + 1] = self.jumps[1:]

        self.jumps = ForwardStar.fill_empty_groups_numba(jumps_expanded)


    @staticmethod
    @njit
    def fill_empty_groups_numba(jumps_expanded):
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


