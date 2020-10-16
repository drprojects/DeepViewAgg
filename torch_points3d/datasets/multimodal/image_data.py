import re
import copy
import logging
import collections

import torch
import torch_geometric
from torch_sparse import coalesce, SparseTensor
from torch_geometric.utils import (contains_isolated_nodes,
                                   contains_self_loops, is_undirected)

from ..utils.num_nodes import maybe_num_nodes

__num_nodes_warn_msg__ = (
    'The number of nodes in your data object can only be inferred by its {} '
    'indices, and hence may result in unexpected batch-wise behavior, e.g., '
    'in case there exists isolated nodes. Please consider explicitly setting '
    'the number of nodes for this data object by assigning it to '
    'data.num_nodes.')


def size_repr(key, item, indent=0):
    indent_str = ' ' * indent
    if torch.is_tensor(item) and item.dim() == 0:
        out = item.item()
    elif torch.is_tensor(item):
        out = str(list(item.size()))
    elif isinstance(item, SparseTensor):
        out = str(item.sizes())[:-1] + f', nnz={item.nnz()}]'
    elif isinstance(item, list) or isinstance(item, tuple):
        out = str([len(item)])
    elif isinstance(item, dict):
        lines = [indent_str + size_repr(k, v, 2) for k, v in item.items()]
        out = '{\n' + ',\n'.join(lines) + '\n' + indent_str + '}'
    elif isinstance(item, str):
        out = f'"{item}"'
    else:
        out = str(item)

    return f'{indent_str}{key}={out}'


class ImageData(object):

    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None,
                 pos=None, normal=None, face=None, **kwargs):
        # add new required attributes, ignore other Data attributes






    def __getitem__(self, key):
        r"""Gets the data of the attribute :obj:`key`."""
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        """Sets the attribute :obj:`key` to :obj:`value`."""
        setattr(self, key, value)

    def __len__(self):
        r"""Returns the number of all present attributes."""
        return len(self.keys)

    def __contains__(self, key):
        r"""Returns :obj:`True`, if the attribute :obj:`key` is present in the
        data."""
        return key in self.keys

    def __iter__(self):
        r"""Iterates over all present attributes in the data, yielding their
        attribute names and content."""
        for key in sorted(self.keys):
            yield key, self[key]



    def __instancecheck__(self, instance):
        return

    def __subclasscheck__(self, subclass):
        return



    def __call__(self, *keys):
        r"""Iterates over all attributes :obj:`*keys` in the data, yielding
        their attribute names and content.
        If :obj:`*keys` is not given this method will iterative over all
        present attributes."""
        for key in sorted(self.keys) if not keys else keys:
            if key in self:
                yield key, self[key]






    def __add__(self, other): 
        """For concatenation with '+' operator 

        only allow if the shared properties are the same:
            same size
            same mask ?
            ...
        """
        return self.string + other 

    def __radd__(self, other):
        """Implements reflected addition."""
        return self.string + other 




    @property
    def num_nodes(self):
        r"""Returns or sets the number of nodes in the graph.

        .. note::
            The number of nodes in your data object is typically automatically
            inferred, *e.g.*, when node features :obj:`x` are present.
            In some cases however, a graph may only be given by its edge
            indices :obj:`edge_index`.
            PyTorch Geometric then *guesses* the number of nodes
            according to :obj:`edge_index.max().item() + 1`, but in case there
            exists isolated nodes, this number has not to be correct and can
            therefore result in unexpected batch-wise behavior.
            Thus, we recommend to set the number of nodes in your data object
            explicitly via :obj:`data.num_nodes = ...`.
            You will be given a warning that requests you to do so.
        """
        if hasattr(self, '__num_nodes__'):
            return self.__num_nodes__
        for key, item in self('x', 'pos', 'normal', 'batch'):
            return item.size(self.__cat_dim__(key, item))
        if hasattr(self, 'adj'):
            return self.adj.size(0)
        if hasattr(self, 'adj_t'):
            return self.adj_t.size(1)
        if self.face is not None:
            logging.warning(__num_nodes_warn_msg__.format('face'))
            return maybe_num_nodes(self.face)
        if self.edge_index is not None:
            logging.warning(__num_nodes_warn_msg__.format('edge'))
            return maybe_num_nodes(self.edge_index)
        return None

    @num_nodes.setter
    def num_nodes(self, num_nodes):
        self.__num_nodes__ = num_nodes





    def debug(self):
        if self.edge_index is not None:
            if self.edge_index.dtype != torch.long:
                raise RuntimeError(
                    ('Expected edge indices of dtype {}, but found dtype '
                     ' {}').format(torch.long, self.edge_index.dtype))

        if self.face is not None:
            if self.face.dtype != torch.long:
                raise RuntimeError(
                    ('Expected face indices of dtype {}, but found dtype '
                     ' {}').format(torch.long, self.face.dtype))

        if self.edge_index is not None:
            if self.edge_index.dim() != 2 or self.edge_index.size(0) != 2:
                raise RuntimeError(
                    ('Edge indices should have shape [2, num_edges] but found'
                     ' shape {}').format(self.edge_index.size()))

        if self.edge_index is not None and self.num_nodes is not None:
            if self.edge_index.numel() > 0:
                min_index = self.edge_index.min()
                max_index = self.edge_index.max()
            else:
                min_index = max_index = 0
            if min_index < 0 or max_index > self.num_nodes - 1:
                raise RuntimeError(
                    ('Edge indices must lay in the interval [0, {}]'
                     ' but found them in the interval [{}, {}]').format(
                         self.num_nodes - 1, min_index, max_index))

        if self.face is not None:
            if self.face.dim() != 2 or self.face.size(0) != 3:
                raise RuntimeError(
                    ('Face indices should have shape [3, num_faces] but found'
                     ' shape {}').format(self.face.size()))

        if self.face is not None and self.num_nodes is not None:
            if self.face.numel() > 0:
                min_index = self.face.min()
                max_index = self.face.max()
            else:
                min_index = max_index = 0
            if min_index < 0 or max_index > self.num_nodes - 1:
                raise RuntimeError(
                    ('Face indices must lay in the interval [0, {}]'
                     ' but found them in the interval [{}, {}]').format(
                         self.num_nodes - 1, min_index, max_index))

        if self.edge_index is not None and self.edge_attr is not None:
            if self.edge_index.size(1) != self.edge_attr.size(0):
                raise RuntimeError(
                    ('Edge indices and edge attributes hold a differing '
                     'number of edges, found {} and {}').format(
                         self.edge_index.size(), self.edge_attr.size()))

        if self.x is not None and self.num_nodes is not None:
            if self.x.size(0) != self.num_nodes:
                raise RuntimeError(
                    ('Node features should hold {} elements in the first '
                     'dimension but found {}').format(self.num_nodes,
                                                      self.x.size(0)))

        if self.pos is not None and self.num_nodes is not None:
            if self.pos.size(0) != self.num_nodes:
                raise RuntimeError(
                    ('Node positions should hold {} elements in the first '
                     'dimension but found {}').format(self.num_nodes,
                                                      self.pos.size(0)))

        if self.normal is not None and self.num_nodes is not None:
            if self.normal.size(0) != self.num_nodes:
                raise RuntimeError(
                    ('Node normals should hold {} elements in the first '
                     'dimension but found {}').format(self.num_nodes,
                                                      self.normal.size(0)))





    def __repr__(self):
        cls = str(self.__class__.__name__)
        has_dict = any([isinstance(item, dict) for _, item in self])

        if not has_dict:
            info = [size_repr(key, item) for key, item in self]
            return '{}({})'.format(cls, ', '.join(info))
        else:
            info = [size_repr(key, item, indent=2) for key, item in self]
            return '{}(\n{}\n)'.format(cls, ',\n'.join(info))