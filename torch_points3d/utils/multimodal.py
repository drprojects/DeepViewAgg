import torch
import numpy as np
import copy


# Key expected to be used for multimodal mappings. Note it is IMPORTANT
# that the key should contain '*index*' to be treated as such by
# torch_geometric's Batch.from_data_list(). This way the point
# indices will be properly updated when stacking multimodal data.
MAPPING_KEY = 'mapping_index'


def tensor_idx(idx):
    """Convert an int, slice, list or numpy index to a torch.LongTensor."""
    if idx is None:
        idx =torch.LongTensor([])
    elif isinstance(idx, int):
        idx = torch.LongTensor([idx])
    elif isinstance(idx, list):
        idx = torch.LongTensor(idx)
    elif isinstance(idx, slice):
        idx = torch.arange(idx.stop)[idx]
    elif isinstance(idx, np.ndarray):
        idx = torch.from_numpy(idx)
    # elif not isinstance(idx, torch.LongTensor):
    #     raise NotImplementedError
    if isinstance(idx, torch.BoolTensor):
        idx = torch.where(idx)[0]
    assert idx.dtype is torch.int64, \
        "Expected LongTensor but got {idx.type} instead."
    # assert idx.shape[0] > 0, \
    #     "Expected non-empty indices. At least one index must be provided."
    return idx


def lexsort(*args, use_cuda=False):
    """Return input tensors sorted in lexicographic order."""
    device = args[0].device
    if device.type == 'cuda':
        use_cuda = True
    elif not torch.cuda.is_available():
        use_cuda = False
    if use_cuda:
        out = cuda_lex_op(*args, op='sort', device=device)
    else:
        out = cpu_lex_op(*args, op='sort', torch_out=True)
    out = [x.to(device) for x in out]
    return out if len(out) > 1 else out[0]


def lexargsort(*args, use_cuda=False):
    """Return indices to sort input tensors in lexicographic order."""
    device = args[0].device
    if device.type == 'cuda':
        use_cuda = True
    elif not torch.cuda.is_available():
        use_cuda = False
    if use_cuda:
        out = cuda_lex_op(*args, op='argsort', device=device)
    else:
        out = cpu_lex_op(*args, op='argsort', torch_out=True)
    return out.to(device)


def lexunique(*args, use_cuda=False):
    """Return unique values in the input tensors sorted in lexicographic
     order."""
    device = args[0].device
    if device.type == 'cuda':
        use_cuda = True
    elif not torch.cuda.is_available():
        use_cuda = False
    if use_cuda:
        out = cuda_lex_op(*args, op='unique', device=device)
    else:
        out = cpu_lex_op(*args, op='unique', torch_out=True)
    out = [x.to(device) for x in out]
    return out if len(out) > 1 else out[0]


def lexargunique(*args, use_cuda=False):
    """Return indices to mapping input tensors to their unique values
    sorted sorted in lexicographic order.
    """
    device = args[0].device
    if device.type == 'cuda':
        use_cuda = True
    elif not torch.cuda.is_available():
        use_cuda = False
    if use_cuda:
        out = cuda_lex_op(*args, op='argunique', device=device)
    else:
        out = cpu_lex_op(*args, op='argunique', torch_out=True)
    return out.to(device)


class CompositeTensor:
    """
    Simple object able to combine the values of a set of 1D int or bool
    tensors of the same shape into a 1D composite torch.Tensor.

    The composite values are built so that they carry the order in
    which the input tensors were passed, which can be used for
    lexicographic-aware operations.
    """

    def __init__(self, *args, device='cpu'):
        """
        Build CompositeTensor from a list of 1D Tensors (or numpy
        arrays).
        """
        supported_formats = (
            torch.int8, torch.int16, torch.int32, torch.int64, torch.bool)
        assert len(args) > 0, "At least one tensor must be provided."
        if ((isinstance(device, str) and device == 'cuda')
            or (isinstance(device, torch.device) and device.type == 'cuda')):
            assert torch.cuda.is_available(), "CUDA not found."

        # Convert input to cuda torch tensor
        tensor_list = [torch.from_numpy(a).to(device)
                       if isinstance(a, np.ndarray)
                       else a.to(device)
                       for a in args]
        assert tensor_list[0].ndim == 1, \
            'Only 1D tensors are accepted as input.'
        assert all([a.shape == tensor_list[0].shape for a in tensor_list]), \
            'All input tensors must have the same shape.'
        assert all([a.dtype in supported_formats for a in tensor_list]), \
            f'All input tensors must be in {supported_formats}. ' \
            f'Received types: {[a.dtype for a in tensor_list]}'

        # Compute the bases to build the composite tensor
        dtype_list = [a.dtype for a in tensor_list]
        dtype_max_list = [torch.iinfo(dt).max for dt in dtype_list]
        dtype_max = max(dtype_max_list)
        dtype = dtype_list[dtype_max_list.index(dtype_max)]
        if tensor_list[0].shape[0] == 0:
            max_list = torch.zeros(len(tensor_list)).long().to(device)
        else:
            max_list = torch.LongTensor([a.abs().max() + 1 for a in tensor_list])
        assert all([torch.prod(max_list) < dtype_max]), \
            'The dtype of at least one of the input tensors must ' \
            'allow the composite computation.'
        base_list = [torch.prod(max_list[i + 1:]).item()
                     for i in range(len(tensor_list) - 1)] + [1]

        # Build the composite tensor
        self.dtype_list = dtype_list
        self.dtype = dtype
        self.max_list = max_list
        self.base_list = base_list
        self.data = sum([
            a.type(dtype) * b for a, b in zip(tensor_list, base_list)])

    @property
    def shape(self):
        return self.data.shape

    @property
    def device(self):
        return self.data.device

    def to(self, device):
        out = copy.deepcopy(self)
        out.data = out.data.to(device)
        return out

    def restore(self):
        """Restore the tensors from the modified composite."""
        out = []
        composite = self.data
        for b, dt in zip(self.base_list, self.dtype_list):
            out.append((composite // b).type(dt))
            composite = composite % b
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.shape}, " \
               f"dtype={self.dtype}, device={self.device})"


class CompositeNDArray:
    """
    Simple object able to combine the values of a set of int or bool 1D
    arrays of the same shape into a 1D composite numpy.ndarray.

    The composite values are built so that they carry the order in
    which the input arrays were passed, which can be used for
    lexicographic-aware operations.
    """

    def __init__(self, *args):
        supported_formats = (np.int8, np.int16, np.int32, np.int64)

        # Convert input to numpy
        array_list = [np.asarray(a.cpu()) if isinstance(a, torch.Tensor)
                      else np.asarray(a)
                      for a in args]
        assert array_list[0].ndim == 1, \
            'Only 1D arrays are accepted as input.'
        assert all([a.shape == array_list[0].shape for a in array_list]), \
            'All input arrays must have the same shape'
        assert all([a.dtype in supported_formats for a in array_list]), \
            f'All input arrays must be in {supported_formats}. ' \
            f'Received types: {[a.dtype in supported_formats for a in array_list]}'

        # Compute the bases to build the composite array
        dtype_list = [a.dtype for a in array_list]
        dtype_max_list = [np.iinfo(dt).max for dt in dtype_list]
        dtype_max = max(dtype_max_list)
        dtype = dtype_list[dtype_max_list.index(dtype_max)]
        if array_list[0].shape[0] == 0:
            max_list = [0 for _ in array_list]
        else:
            max_list = [np.abs(a).max() + 1 for a in array_list]
        assert all([np.prod(max_list) < dtype_max]), \
            'The dtype of at least one of the input arrays must ' \
            'allow the composite computation.'
        base_list = [np.prod(max_list[i + 1:])
                     for i in range(len(array_list) - 1)] + [1]

        # Build the composite array
        self.dtype_list = dtype_list
        self.dtype = dtype
        self.max_list = max_list
        self.base_list = base_list
        self.data = sum([a.astype(dtype) * b
                         for a, b in zip(array_list, base_list)])

    @property
    def shape(self):
        return self.data.shape

    def restore(self, torch_out=False):
        """Restore the arrays from the modified composite."""
        out = []
        composite = self.data
        for b, dt in zip(self.base_list, self.dtype_list):
            out.append((composite // b).astype(dt))
            composite = composite % b

        # Convert to torch Tensor if need be
        if torch_out:
            out = [torch.from_numpy(o) for o in out]

        return out

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.shape}, " \
               f"dtype={self.dtype})"


def cuda_lex_op(*args, op='unique', device='cuda'):
    """
    Lexicographic-aware operations on a set of input tensors on GPU.

    Fastest way of performing `sort`, `argsort`, `unique`, `argunique`
    operations on large (>10^6) tensors on GPU, provided that CUDA is
    available.

    Returns torch cuda tensors.
    """
    # Create a composite cuda tensor holding the input data
    composite = CompositeTensor(*args, device=device)

    # Core operation on the composite tensor
    if op == 'unique':
        composite.data = torch.unique(composite.data, sorted=True)
    elif op == 'argunique':
        # NB: unlike numpy.unique, the returned index has the same size
        # as the input array. Further processing torch.scatter is
        # needed to isolate single occurrences for each unique value.
        unique, inverse = torch.unique(
            composite.data, sorted=True, return_inverse=True)
        perm = torch.arange(
            inverse.size(0), dtype=inverse.dtype, device=device)
        inverse, perm = inverse.flip([0]), perm.flip([0])
        return inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
    elif op == 'sort':
        composite.data = torch.sort(composite.data).values
    elif op == 'argsort':
        return torch.sort(composite.data).indices
    else:
        raise NotImplementedError

    return composite.restore()


def cpu_lex_op(*args, op='unique', torch_out=True):
    """
    Lexicographic-aware operations on a set of input arrays on CPU.

    Fastest way of performing `sort`, `argsort`, `unique`, `argunique`
    operations on large (>10^6) arrays on CPU, if CUDA is not available.

    Remark: as of 20.12.2020 Numba is slower than Numpy on sorting
    operations.

    Returns either numpy ndarrays or torch cpu tensors, depending on
    `torch_out`.
    """
    # Create a composite cuda tensor holding the input data
    composite = CompositeNDArray(*args)

    # Core operation on the composite array
    if op == 'unique':
        composite.data = np.unique(composite.data)
    elif op == 'argunique':
        idx = np.unique(composite.data, return_index=True)[1]
        if torch_out:
            idx = torch.from_numpy(idx)
        return idx
    elif op == 'sort':
        composite.data.sort()  # in-place sort is faster
    elif op == 'argsort':
        idx = np.argsort(composite.data)
        if torch_out:
            idx = torch.from_numpy(idx)
        return idx
    else:
        raise NotImplementedError

    return composite.restore(torch_out=torch_out)


"""
from torch_points3d.utils.multimodal import *
import numpy as np
import torch
import torch_scatter
from time import time

n_points = 10**7
i_max = 10**6
dim = 3
a_np = np.random.randint(0, high=i_max, size=(n_points, dim))
a_torch = torch.from_numpy(a_np).cuda()

args_np = [np.squeeze(a) for a in np.split(a_np, a_np.shape[1], axis=1)]
args_torch = [a.squeeze() for a in a_torch.split(1, dim=1)]

def run_test(f, input):
    start = time()
    out_cuda = f(*input, device='cuda')
    t_cuda = time() - start
    start = time()
    out_cpu = f(*input, device='cpu')
    t_cpu = time() - start
    print_times(t_cuda, t_cpu)
    return out_cuda, out_cpu

def print_times(t_cuda, t_cpu):
    print(f"CUDA: {t_cuda * 1000:0.0f} ms    CPU: {t_cpu * 1000:0.0f} ms    speedup: {t_cpu / t_cuda:0.2f}")

print("lexsort")
out_cuda, out_cpu = run_test(lexsort, args_np)
print(f"np check: {(out_cuda[-1].cpu() == out_cpu[-1]).all()}")
out_cuda, out_cpu = run_test(lexsort, args_torch)
print(f"torch check: {(out_cuda[-1].cpu() == out_cpu[-1]).all()}\n")

print("lexargsort")
out_cuda, out_cpu = run_test(lexargsort, args_np)
print(f"np check: {(args_np[-1][out_cuda.cpu()] == args_np[-1][out_cpu]).all()}")
out_cuda, out_cpu = run_test(lexargsort, args_torch)
print(f"torch check: {(args_torch[-1][out_cuda.cpu()] == args_torch[-1][out_cpu]).all()}\n")

print("lexunique")
out_cuda, out_cpu = run_test(lexunique, args_np)
print(f"np check: {(out_cuda[-1].cpu() == out_cpu[-1]).all()}")
out_cuda, out_cpu = run_test(lexunique, args_torch)
print(f"torch check: {(out_cuda[-1].cpu() == out_cpu[-1]).all()}\n")

print("lexargunique")
out_cuda, out_cpu = run_test(lexargunique, args_np)
print(f"np check: {(args_np[-1][out_cuda.cpu()] == args_np[-1][out_cpu]).all()}")
out_cuda, out_cpu = run_test(lexargunique, args_torch)
print(f"torch check: {(args_torch[-1][out_cuda.cpu()] == args_torch[-1][out_cpu]).all()}\n")

"""

