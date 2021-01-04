import torch
import numpy as np


def composite_operation(*args, op='unique', torch_out=False):
    """
    Compute core operations on a composite array built from input 1D arrays.

    This is typically useful as faster replacement for `np.sort(..., axis=0)`,
    `np.argsort(..., axis=0)`, `np.lexsort(..., axis=0)` or
    `np.unique(..., axis=0)` on large (>10^6) arrays.

    In this regard, torch is even slower than numpy, so this function supports
    `torch.Tensor` inputs and outputs with `torch_out=True`.

    Remark: as of 20.12.2020 Numba is slower than Numpy on sorting operations.
    """
    # Convert input to numpy
    array_list = [np.asarray(a) for a in args]
    assert array_list[0].ndim == 1,\
        'Only 1D arrays are accepted as input.'
    assert all([a.shape == array_list[0].shape for a in array_list]),\
        'All input arrays must have the same shape'

    # Compute the bases to build the composite array
    dtype_list = [a.dtype for a in array_list]
    dtype_max = max([np.iinfo(dt).max for dt in dtype_list])
    max_list = [a.max()+1 for a in array_list]
    assert all([np.prod(max_list) < dtype_max]), \
        'The dtype of at least one of the input arrays must allow the composite computation.'
    base_list = [np.prod(max_list[i+1:]) for i in range(len(array_list)-1)] + [1]

    # Build the composite array
    composite = sum([a * b for a, b in zip(array_list, base_list)])

    # Core operation on the composite array
    if op == 'unique':
        composite = np.unique(composite)
    elif op == 'sort':
        composite.sort()  # in-place sort is faster
    elif op == 'argsort':
        idx = np.argsort(composite)
        if torch_out:
            idx = torch.from_numpy(idx)
            composite = torch.from_numpy(composite)
        return idx, composite
    else:
        raise NotImplementedError

    # Restore the arrays from the modified composite
    out_list = []
    for b, dt in zip(base_list, dtype_list):
        out_list.append((composite // b).astype(dt))
        composite = composite % b

    # Convert to torch Tensor if need be
    if torch_out:
        out_list = [torch.from_numpy(o) for o in out_list]

    return out_list


