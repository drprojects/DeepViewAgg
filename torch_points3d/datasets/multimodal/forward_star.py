import numpy as np
from numba import njit


def compute_jumps(indices):
    """
    Convert dense format to forward star. Indices are assumed to be already
    sorted, if sorting is necessary. 
    """
    assert len(indices.shape) == 1
    return from_dense_numba(indices)


@njit
def compute_jumps_numba(indices):
    # Compute the jumps
    idx_previous = indices[0]
    jumps = []
    for i, idx in enumerate(indices):
        if idx != idx_previous:
            jumps.append(i)
            idx_previous = idx

    # Last index must be treated separately
    jumps.append(i+1)

    return jumps


def all_jumps(jumps, indices_sorted, idx_max=None):
    """
    Produce the jumps for the complete array of indices range(0, idx_max).

    Indices are assumed to be sorted and to correspond to the provided jumps.
    The indices with no jump will have 0-size jumps in the returned jumps. 
    """
    assert jumps.shape[0] == indices.shape[0]
    assert (np.array_equal(np.sort(indices_sorted), indices_sorted),
        "Indices are not sorted.")
    if idx_max:
        assert idx_max >= indices_sorted.max()
    else:
        idx_max = indices_sorted.max()

    jumps_complete = np.zeros(idx_max + 1, dtype=indices_sorted.dtype)
    jumps_complete[indices_sorted] = jumps

    return all_jumps_numba(jumps_complete)


@njit
def all_jumps_numba(jumps_complete):
    jump_previous = 0
    for i in range(jumps_complete.shape[0]):
        if jumps_complete[i] < jump_previous:
            jumps_complete[i] = jump_previous
        jump_previous = jumps_complete[i]
    return jumps_complete
