from typing import List, Optional
from tqdm.auto import tqdm as tq
import itertools
import numpy as np
import math
import re
import torch
import random
from torch.nn import functional as F
from functools import partial
from joblib import Parallel, delayed
import math
import functools
from pykeops.torch import LazyTensor

from torch_geometric.nn import fps, radius, knn, voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.pool import pool_pos, pool_batch
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.data import Data, Batch
from torch_points3d.datasets.multiscale_data import MultiScaleData
from torch_points3d.utils.transform_utils import SamplingStrategy
from torch_points3d.utils.config import is_list
from torch_points3d.utils import is_iterable
from torch_points3d.utils.geometry import euler_angles_to_rotation_matrix
from torch_points3d.core.spatial_ops.neighbour_finder import \
    RadiusNeighbourFinder, FAISSGPUKNNNeighbourFinder


class Random3AxisRotation(object):
    """
    Rotate pointcloud with random angles along x, y, z axis

    The angles should be given `in degrees`.

    Parameters
    -----------
    apply_rotation: bool:
        Whether to apply the rotation
    rot_x: float
        Rotation angle in degrees on x axis
    rot_y: float
        Rotation anglei n degrees on y axis
    rot_z: float
        Rotation angle in degrees on z axis
    """

    def __init__(self, apply_rotation: bool = True, rot_x: float = None, rot_y: float = None, rot_z: float = None):
        self._apply_rotation = apply_rotation
        if apply_rotation:
            if (rot_x is None) and (rot_y is None) and (rot_z is None):
                raise Exception("At least one rot_ should be defined")

        self._rot_x = np.abs(rot_x) if rot_x else 0
        self._rot_y = np.abs(rot_y) if rot_y else 0
        self._rot_z = np.abs(rot_z) if rot_z else 0

        self._degree_angles = [self._rot_x, self._rot_y, self._rot_z]

    def generate_random_rotation_matrix(self):
        thetas = torch.zeros(3, dtype=torch.float)
        for axis_ind, deg_angle in enumerate(self._degree_angles):
            if deg_angle > 0:
                rand_deg_angle = random.random() * 2 * deg_angle - deg_angle
                rand_radian_angle = float(rand_deg_angle * np.pi) / 180.0
                thetas[axis_ind] = rand_radian_angle
        return euler_angles_to_rotation_matrix(thetas, random_order=True)

    def __call__(self, data):
        if self._apply_rotation:
            pos = data.pos.float()
            M = self.generate_random_rotation_matrix()
            data.pos = pos @ M.T
            if getattr(data, "norm", None) is not None:
                data.norm = data.norm.float() @ M.T
        return data

    def __repr__(self):
        return "{}(apply_rotation={}, rot_x={}, rot_y={}, rot_z={})".format(
            self.__class__.__name__, self._apply_rotation, self._rot_x, self._rot_y, self._rot_z
        )


class RandomTranslation(object):
    """
    random translation
    Parameters
    -----------
    delta_min: list
        min translation
    delta_max: list
        max translation
    """

    def __init__(self, delta_max: List = [1.0, 1.0, 1.0], delta_min: List = [-1.0, -1.0, -1.0]):
        self.delta_max = torch.tensor(delta_max)
        self.delta_min = torch.tensor(delta_min)

    def __call__(self, data):
        pos = data.pos
        trans = torch.rand(3) * (self.delta_max - self.delta_min) + self.delta_min
        data.pos = pos + trans
        return data

    def __repr__(self):
        return "{}(delta_min={}, delta_max={})".format(self.__class__.__name__, self.delta_min, self.delta_max)


class AddFeatsByKeys(object):
    """This transform takes a list of attributes names and if allowed, add them to x

    Example:

        Before calling "AddFeatsByKeys", if data.x was empty

        - transform: AddFeatsByKeys
          params:
              list_add_to_x: [False, True, True]
              feat_names: ['normal', 'rgb', "elevation"]
              input_nc_feats: [3, 3, 1]

        After calling "AddFeatsByKeys", data.x contains "rgb" and "elevation". Its shape[-1] == 4 (rgb:3 + elevation:1)
        If input_nc_feats was [4, 4, 1], it would raise an exception as rgb dimension is only 3.

    Paremeters
    ----------
    list_add_to_x: List[bool]
        For each boolean within list_add_to_x, control if the associated feature is going to be concatenated to x
    feat_names: List[str]
        The list of features within data to be added to x
    input_nc_feats: List[int], optional
        If provided, evaluate the dimension of the associated feature shape[-1] found using feat_names and this provided value. It allows to make sure feature dimension didn't change
    stricts: List[bool], optional
        Recommended to be set to list of True. If True, it will raise an Exception if feat isn't found or dimension doesn t match.
    delete_feats: List[bool], optional
        Whether we want to delete the feature from the data object. List length must match teh number of features added.
    """

    def __init__(
            self,
            list_add_to_x: List[bool],
            feat_names: List[str],
            input_nc_feats: List[Optional[int]] = None,
            stricts: List[bool] = None,
            delete_feats: List[bool] = None,
    ):

        self._feat_names = feat_names
        self._list_add_to_x = list_add_to_x
        self._delete_feats = delete_feats
        if self._delete_feats:
            assert len(self._delete_feats) == len(self._feat_names)
        from torch_geometric.transforms import Compose

        num_names = len(feat_names)
        if num_names == 0:
            raise Exception("Expected to have at least one feat_names")

        assert len(list_add_to_x) == num_names

        if input_nc_feats:
            assert len(input_nc_feats) == num_names
        else:
            input_nc_feats = [None for _ in range(num_names)]

        if stricts:
            assert len(stricts) == num_names
        else:
            stricts = [True for _ in range(num_names)]

        transforms = [
            AddFeatByKey(add_to_x, feat_name, input_nc_feat=input_nc_feat, strict=strict)
            for add_to_x, feat_name, input_nc_feat, strict in zip(list_add_to_x, feat_names, input_nc_feats, stricts)
        ]

        self.transform = Compose(transforms)

    def _process(self, data: Data):
        data = self.transform(data)
        if self._delete_feats:
            for feat_name, delete_feat in zip(self._feat_names, self._delete_feats):
                if delete_feat:
                    delattr(data, feat_name)
        return data
    
    def __call__(self, data):
        if isinstance(data, list):
            data = [self._process(d) for d in data]
        else:
            data = self._process(data)
        return data

    def __repr__(self):
        msg = ""
        for f, a in zip(self._feat_names, self._list_add_to_x):
            msg += "{}={}, ".format(f, a)
        return "{}({})".format(self.__class__.__name__, msg[:-2])


class AddFeatByKey(object):
    """This transform is responsible to get an attribute under feat_name and add it to x if add_to_x is True

    Parameters
    ----------
    add_to_x: bool
        Control if the feature is going to be added/concatenated to x
    feat_name: str
        The feature to be found within data to be added/concatenated to x
    input_nc_feat: int, optional
        If provided, check if feature last dimension maches provided value.
    strict: bool, optional
        Recommended to be set to True. If False, it won't break if feat isn't found or dimension doesn t match. (default: ``True``)
    """

    def __init__(self, add_to_x, feat_name, input_nc_feat=None, strict=True):

        self._add_to_x: bool = add_to_x
        self._feat_name: str = feat_name
        self._input_nc_feat = input_nc_feat
        self._strict: bool = strict

    def _process(self, data: Data):
        if not self._add_to_x:
            return data
        feat = getattr(data, self._feat_name, None)
        if feat is None:
            if self._strict:
                raise Exception("Data should contain the attribute {}".format(self._feat_name))
            else:
                return data
        else:
            if self._input_nc_feat:
                feat_dim = 1 if feat.dim() == 1 else feat.shape[-1]
                if self._input_nc_feat != feat_dim and self._strict:
                    raise Exception("The shape of feat: {} doesn t match {}".format(feat.shape, self._input_nc_feat))
            x = getattr(data, "x", None)
            if x is None:
                if self._strict and data.pos.shape[0] != feat.shape[0]:
                    raise Exception("We expected to have an attribute x")
                if feat.dim() == 1:
                    feat = feat.unsqueeze(-1)
                data.x = feat
            else:
                if x.shape[0] == feat.shape[0]:
                    if x.dim() == 1:
                        x = x.unsqueeze(-1)
                    if feat.dim() == 1:
                        feat = feat.unsqueeze(-1)
                    data.x = torch.cat([x, feat], dim=-1)
                else:
                    raise Exception(
                        "The tensor x and {} can't be concatenated, x: {}, feat: {}".format(
                            self._feat_name, x.pos.shape[0], feat.pos.shape[0]
                        )
                    )
        return data

    def __call__(self, data):
        if isinstance(data, list):
            data = [self._process(d) for d in data]
        else:
            data = self._process(data)
        return data
    
    def __repr__(self):
        return "{}(add_to_x: {}, feat_name: {}, strict: {})".format(
            self.__class__.__name__, self._add_to_x, self._feat_name, self._strict
        )


def compute_planarity(eigenvalues):
    r"""
    compute the planarity with respect to the eigenvalues of the covariance matrix of the pointcloud
    let
    :math:`\lambda_1, \lambda_2, \lambda_3` be the eigenvalues st:

    .. math::
        \lambda_1 \leq \lambda_2 \leq \lambda_3

    then planarity is defined as:

    .. math::
        planarity = \frac{\lambda_2 - \lambda_1}{\lambda_3}
    """

    return (eigenvalues[1] - eigenvalues[0]) / eigenvalues[2]


class NormalFeature(object):
    """
    add normal as feature. if it doesn't exist, compute normals
    using PCA
    """

    def __call__(self, data):
        if getattr(data, "norm", None) is None:
            raise NotImplementedError("TODO: Implement normal computation")

        norm = data.norm
        if data.x is None:
            data.x = norm
        else:
            data.x = torch.cat([data.x, norm], -1)
        return data


def batch_pca(xyz):
    """
    Compute the PCA of a batch of point clouds of size (*, N, M).
    """
    assert 2 <= xyz.dim() <= 3
    xyz = xyz.unsqueeze(0) if xyz.dim() == 2 else xyz

    pos_centered = xyz - xyz.mean(dim=1).unsqueeze(1)
    cov_matrix = pos_centered.transpose(1, 2).bmm(pos_centered) / pos_centered.shape[1]
    eval, evec = torch.symeig(cov_matrix, eigenvectors=True)

    # If Nan values are computed, return equal eigenvalues and
    # Identity eigenvectors
    idx_nan = torch.where(torch.logical_and(
        eval.isnan().any(1), evec.flatten(1).isnan().any(1)))
    eval[idx_nan] = torch.ones(3, dtype=eval.dtype, device=xyz.device)
    evec[idx_nan] = torch.eye(3, dtype=evec.dtype, device=xyz.device)

    # Precision errors may cause close-to-zero eigenvalues to be
    # negative. Hard-code these to zero
    eval[torch.where(eval < 0)] = 0

    return eval, evec


class PCACompute(object):
    r"""
    compute `Principal Component Analysis <https://en.wikipedia.org/wiki/Principal_component_analysis>`__ of a point cloud :math:`x_1,\dots, x_n`.
    It computes the eigenvalues and the eigenvectors of the matrix :math:`C` which is the covariance matrix of the point cloud:

    .. math::
        x_{centered} &= \frac{1}{n} \sum_{i=1}^n x_i

        C &= \frac{1}{n} \sum_{i=1}^n (x_i - x_{centered})(x_i - x_{centered})^T

    store the eigen values and the eigenvectors in data.
    in eigenvalues attribute and eigenvectors attributes.
    data.eigenvalues is a tensor :math:`(\lambda_1, \lambda_2, \lambda_3)` such that :math:`\lambda_1 \leq \lambda_2 \leq \lambda_3`.

    data.eigenvectors is a 3 x 3 matrix such that the column are the eigenvectors associated to their eigenvalues
    Therefore, the first column of data.eigenvectors estimates the normal at the center of the pointcloud.
    """

    def __call__(self, data):
        eval, evec = batch_pca(data.pos)
        data.eigenvalues = eval.squeeze(0)
        data.eigenvectors = evec.squeeze(0)
        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class PCAComputePointwise(object):
    """
    Compute PCA for the local neighborhood of each point in the cloud.

    Input data is expected to be stored in DENSE format.

    Results are saved in `eigenvalues` and `eigenvectors` attributes.
    `data.eigenvalues` is a tensor
    :math:`(\lambda_1, \lambda_2, \lambda_3)` such that
    :math:`\lambda_1 \leq \lambda_2 \leq \lambda_3`.
    `data.eigenvectors` is 1x9 tensor containing the eigenvectors
    associated with `data.eigenvalues`, concatenated in the same order.

    Parameters
    ----------
    num_neighbors: int, optional
        Controls the maximum number of neighbors on which to compute
        PCA. If `r=None`, `num_neighbors` will be used as K for
        K-nearest neighbor search. Otherwise, `num_neighbors` will be
        the maximum number of neighbors used in radial neighbor search.
    r: float, optional
        If not `None`, neighborhoods will be computed with a
        radius-neighbor approach. If `None`, K-nearest neighbors will
        be used.
    use_full_pos: bool, optional
        If True, the neighborhood search will be carried on the point
        positions found in the `data.full_pos`. An error will be raised
        if data carries no such attribute. See `GridSampling3D` for
        producing `data.full_pos`.
        If False, the neighbor search will be computed on `data.pos`.
    use_cuda: bool, optional
        If True, the computation will be carried on CUDA.
    workers: int, optional
        If not `None`, the features computation will be distributed
        across the provided number of workers.
    """

    def __init__(
            self, num_neighbors=40, r=None, use_full_pos=False, use_cuda=False,
            use_faiss=True, ncells=None, nprobes=10, chunk_size=1000000):
        self.num_neighbors = num_neighbors
        self.r = r
        self.use_full_pos = use_full_pos
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.use_faiss = use_faiss and torch.cuda.is_available()
        self.ncells = ncells
        self.nprobes = nprobes
        self.chunk_size = chunk_size

    def _process(self, data: Data):
        assert getattr(data, 'pos', None) is not None, \
            "Data must contain a 'pos' attribute."
        assert not self.use_full_pos \
               or getattr(data, 'full_pos', None) is not None, \
            "Data must contain a 'full_pos' attribute."

        # Recover the query and search clouds
        xyz_query = data.pos
        xyz_search = data.full_pos if self.use_full_pos else data.pos

        # Move computation to CUDA if required
        input_device = xyz_query.device
        if self.use_cuda and not xyz_query.is_cuda and not self.use_faiss:
            xyz_query = xyz_query.cuda()
            xyz_search = xyz_search.cuda()

        # Compute the neighborhoods
        if self.r is not None:
            # Radius-NN search with torch_points_kernel
            sampler = RadiusNeighbourFinder(
                self.r, self.num_neighbors, conv_type='DENSE')
            neighbors = sampler.find_neighbours(
                xyz_search.unsqueeze(0), xyz_query.unsqueeze(0))[0]
        elif self.use_faiss:
            # K-NN search with FAISS
            nn_finder = FAISSGPUKNNNeighbourFinder(
                self.num_neighbors, ncells=self.ncells, nprobes=self.nprobes)
            neighbors = nn_finder(xyz_search, xyz_query, None, None)
        else:
            # K-NN search with KeOps. If the number of points is greater
            # than 16 millions, KeOps requires double precision.
            xyz_query = xyz_query.contiguous()
            xyz_search = xyz_search.contiguous()
            if xyz_search.shape[0] > 1.6e7:
                xyz_query_keops = LazyTensor(xyz_query[:, None, :].double())
                xyz_search_keops = LazyTensor(xyz_search[None, :, :].double())
            else:
                xyz_query_keops = LazyTensor(xyz_query[:, None, :])
                xyz_search_keops = LazyTensor(xyz_search[None, :, :])
            d_keops = ((xyz_query_keops - xyz_search_keops) ** 2).sum(dim=2)
            neighbors = d_keops.argKmin(self.num_neighbors, dim=1)
            # raise NotImplementedError(
            #     "Fast K-NN search has not been implemented yet. Please "
            #     "consider using radius search instead.")

        # Compute PCA for each neighborhood
        # Note: this is surprisingly slow on GPU, so better run on CPU
        eigenvalues = []
        eigenvectors = []
        n_chunks = math.ceil(neighbors.shape[0] / self.chunk_size)
        for i in range(n_chunks):
            xyz_neigh_batch = xyz_search[
                neighbors[i * self.chunk_size: (i + 1) * self.chunk_size]]
            eval, evec = batch_pca(xyz_neigh_batch.cpu())
            evec = evec.transpose(2, 1).flatten(1)
            eigenvalues.append(eval)
            eigenvectors.append(evec)
        eigenvalues = torch.cat(eigenvalues, dim=0)
        eigenvectors = torch.cat(eigenvectors, dim=0)

        # Save eigendecomposition results in data attributes
        data.eigenvalues = eigenvalues.to(input_device)
        data.eigenvectors = eigenvectors.to(input_device)

        return data

    def __call__(self, data):
        if isinstance(data, list):
            data = [self._process(d) for d in tq(data)]
        else:
            data = self._process(data)
        return data

    def __repr__(self):
        attr_repr = ', '.join([f'{k}={v}' for k, v in self.__dict__.items()])
        return f'{self.__class__.__name__}({attr_repr})'


class EigenFeatures(object):
    """
    Compute local geometric features based on local eigenvalues and
    eigenvectors.

    The following local geometric features are computed and saved in
    dedicated data attributes: `norm`, `scattering`, `linearity` and
    `planarity`. The formulation of those is inspired from
    "Hierarchical extraction of urban objects from mobile laser
    scanning data" [Yang et al. 2015]

    Data is expected to carry `eigenvectors` and `eigenvectors`
    attributes:
    `data.eigenvalues` is a tensor
    :math:`(\lambda_1, \lambda_2, \lambda_3)` such that
    :math:`\lambda_1 \leq \lambda_2 \leq \lambda_3`.
    `data.eigenvectors` is 1x9 tensor containing the eigenvectors
    associated with `data.eigenvalues`, concatenated in the same order.
    See `PCAComputePointwise` for generating those.

    Parameters
    ----------
    norm: bool, optional
        If True, the normal to the local surface will be computed.
    linearity: bool, optional
        If True, the local linearity will be computed.
    planarity: bool, optional
        If True, the local planarity will be computed.
    scattering: bool, optional
        If True, the local scattering will be computed.
    temperature: float, optional
        If set to a float value, the returned features will be run
        through a scaled softmax with temperature being the scale. Set
        to None by default.
    """

    def __init__(self, norm=True, linearity=True, planarity=True,
                 scattering=True, temperature=None):
        self.norm = norm
        self.linearity = linearity
        self.planarity = planarity
        self.scattering = scattering
        self.temperature = temperature

    def _process(self, data: Data):
        assert getattr(data, 'eigenvalues', None) is not None, \
            "Data must contain an 'eigenvalues' attribute."
        assert getattr(data, 'eigenvectors', None) is not None, \
            "Data must contain an 'eigenvectors' attribute."

        if self.norm:
            # The normal is the eigenvector carried by the smallest
            # eigenvalue
            data.norm = data.eigenvectors[:, :3]

        # Eigenvalues: 0 <= l0 <= l1 <= l2
        # Following, [Yang et al. 2015] we use the sqrt of eigenvalues
        v0 = data.eigenvalues[:, 0].sqrt().squeeze()
        v1 = data.eigenvalues[:, 1].sqrt().squeeze()
        v2 = data.eigenvalues[:, 2].sqrt().squeeze() + 1e-6

        # Compute the eigen features
        linearity = (v2 - v1) / v2
        planarity = (v1 - v0) / v2
        scattering = v0 / v2

        # Compute the softmax version of the features, for more
        # opinionated geometric information. As a heuristic, set
        # temperature=5 for clouds of 30 points or more.
        if self.temperature:
            values = (self.temperature * torch.cat([
                linearity.view(-1, 1),
                planarity.view(-1, 1),
                scattering.view(-1, 1)], dim=1)).exp()
            values = values / values.sum(dim=1).view(-1, 1)
            linearity = values[:, 0]
            planarity = values[:, 1]
            scattering = values[:, 2]

        if self.linearity:
            data.linearity = linearity

        if self.planarity:
            data.planarity = planarity

        if self.scattering:
            data.scattering = scattering

        return data

    def __call__(self, data):
        if isinstance(data, list):
            data = [self._process(d) for d in data]
        else:
            data = self._process(data)
        return data

    def __repr__(self):
        attr_repr = ', '.join([f'{k}={v}' for k, v in self.__dict__.items()])
        return f'{self.__class__.__name__}({attr_repr})'


class AddOnes(object):
    """
    Add ones tensor to data
    """

    def __call__(self, data):
        num_nodes = data.pos.shape[0]
        data.ones = torch.ones((num_nodes, 1)).float()
        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class XYZFeature(object):
    """
    Add the X, Y and Z as a feature
    Parameters
    -----------
    add_x: bool [default: False]
        whether we add the x position or not
    add_y: bool [default: False]
        whether we add the y position or not
    add_z: bool [default: True]
        whether we add the z position or not
    """

    def __init__(self, add_x=False, add_y=False, add_z=True):
        self._axis = []
        axis_names = ["x", "y", "z"]
        if add_x:
            self._axis.append(0)
        if add_y:
            self._axis.append(1)
        if add_z:
            self._axis.append(2)

        self._axis_names = [axis_names[idx_axis] for idx_axis in self._axis]

    def __call__(self, data):
        assert data.pos is not None
        for axis_name, id_axis in zip(self._axis_names, self._axis):
            f = data.pos[:, id_axis].clone()
            setattr(data, "pos_{}".format(axis_name), f)
        return data

    def __repr__(self):
        return "{}(axis={})".format(self.__class__.__name__, self._axis_names)
