import numpy as np
from numba import njit
import torch
import torch_scatter
from PIL import Image
from pykeops.torch import LazyTensor
from abc import ABC, abstractmethod
from torch_points3d.utils.multimodal import torch_to_numba


# -------------------------------------------------------------------- #
#                              Visibility                              #
# -------------------------------------------------------------------- #

class VisibilityModel(ABC):

    def __init__(
            self, crop_top=0, crop_bottom=0, r_max=30, r_min=0.5):
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
        self.r_max = r_max
        self.r_min = r_min

    def __call__(self, xyz, img_xyz, img_extrinsic, camera, **kwargs):
        """Compute the visibility of a point cloud with respect to a
        given camera pose.

        :param xyz:
        :param img_xyz:
        :param img_extrinsic:
        :param camera:
        :return:
        """
        device = xyz.device

        # Compute camera projection
        idx_1, dist, x_proj, y_proj = camera.shoot(
            xyz, img_xyz, img_extrinsic, crop_top=self.crop_top,
            crop_bottom=self.crop_bottom, r_max=self.r_max, r_min=self.r_min)

        # Return if no projections are found
        if x_proj.shape[0] == 0:
            out = {}
            out['idx'] = torch.empty((0,), dtype=torch.long, device=device)
            out['x'] = torch.empty((0,), dtype=torch.long, device=device)
            out['y'] = torch.empty((0,), dtype=torch.long, device=device)
            out['depth'] = torch.empty((0,), dtype=torch.float, device=device)
            return out

        # Compute visibility of projected points
        idx_2, x_pix, y_pix = self._run(
            x_proj, y_proj, dist, xyz, img_extrinsic, camera, **kwargs)

        # Keep data only for mapped points
        idx = idx_1[idx_2]
        dist = dist[idx_2]

        out = {}
        out['idx'] = idx
        out['x'] = x_pix
        out['y'] = y_pix
        out['depth'] = dist

        return out

    @abstractmethod
    def _run(self, x_proj, y_proj, dist, xyz, img_extrinsic, camera, **kwargs):
        pass

    def __repr__(self):
        attr_repr = ', '.join([f'{k}={v}' for k, v in self.__dict__.items()])
        return f'{self.__class__.__name__}({attr_repr})'


# -------------------------------------------------------------------- #
#                     Visibility Method - Splatting                    #
# -------------------------------------------------------------------- #

class SplattingVisibility(VisibilityModel, ABC):
    """Compute visibility model by considering points as voxels (cubes)
    facing the camera. The projection of each of these voxels in the
    pixel grid is called a splat. The occlusions are recovered by
    rendering these splats using a z-buffering approach on the voxel
    distance.

    The voxels can be artificially swollen to tune the occlusions for
    voxels located close to the camera.

    The returned mapping may either associate visible points with the mask
    of the visible portion of their splat, or simply with the point's
    projection only.
    """

    def __init__(
            self, crop_top=0, crop_bottom=0, r_max=30, r_min=0.5,
            voxel=0.1, k_swell=1.0, d_swell=1000, exact=False):
        super().__init__(
            crop_top=crop_top, crop_bottom=crop_bottom, r_max=r_max,
            r_min=r_min)
        self.voxel = voxel
        self.k_swell = k_swell
        self.d_swell = d_swell
        self.exact = exact

    def _run(self, x_proj, y_proj, dist, xyz, img_extrinsic, camera, **kwargs):
        # Let the camera compute the voxel splats
        splat = camera.splat(
            x_proj, y_proj, dist, xyz, img_extrinsic, crop_top=self.crop_top,
            crop_bottom=self.crop_bottom, voxel=self.voxel,
            k_swell=self.k_swell, d_swell=self.d_swell)

        # Z-buffering of the rectangular splats
        indices, x_pix, y_pix = z_buffering(
            splat, x_proj, y_proj, dist, exact=self.exact)

        return indices, x_pix, y_pix


def bbox_to_xy_grid_cuda(bbox):
    """Convert a tensor of bounding boxes pixel coordinates to tensors
    of x and y pixel coordinates accounting for all pixels in the
    bounding boxes. This is typically used to convert splats into pixel
    coordinates.

    Bounding boxes are expected to be a torch.LongTensor of shape (N, 4)
    with columns accounting for x_a, x_b, y_a, y_b coordinates,
    respectively.

    This would be the vectorized equivalent of:
    `
    x_range_list = [torch.arange(x_a, x_b) for x_a, x_b in bbox[:, [0, 1]]]
    y_range_list = [torch.arange(y_a, y_b) for y_a, y_b in bbox[:, [2, 3]]]
    grids = [torch.meshgrid(x_range, y_range)
             for x_range, y_range in zip(x_range_list, y_range_list)]
    x = torch.cat([g[0].flatten() for g in grids])
    y = torch.cat([g[1].flatten() for g in grids])
    `

    :param bbox:
    :return:
    """
    assert bbox.dim() == 2 and bbox.shape[0] > 0 and bbox.shape[1] == 4
    assert not bbox.is_floating_point(), \
        f"bbox should be an int32 or int64 tensor but received " \
        f"bbox.dtype={bbox.dtype} instead."

    # Initializations
    device = bbox.device
    zero = torch.zeros(1, device=device, dtype=torch.long)
    bbox = bbox.long()

    # Recover the bounding boxes' starting x and y coordinates, widths
    # and total number of pixels
    box_x = bbox[:, 0]
    box_width_x = bbox[:, 1] - bbox[:, 0]
    box_y = bbox[:, 2]
    box_width_y = bbox[:, 3] - bbox[:, 2]
    n_pix = (box_width_x * box_width_y).sum()

    # Build the x coordinates
    off_1 = torch.cat((zero, box_width_x[:-1].cumsum(0))).repeat_interleave(
        box_width_x)
    off_2 = box_x.repeat_interleave(box_width_x)
    out = torch.arange(box_width_x.sum(), device=device) - off_1 + off_2
    x = out.repeat_interleave(box_width_y.repeat_interleave(box_width_x))

    # Build the y coordinates
    off_per_group = torch.cat(
        (zero, box_width_y.repeat_interleave(box_width_x).cumsum(0)))
    off_1 = off_per_group[:-1].repeat_interleave(
        off_per_group[1:] - off_per_group[:-1])
    off_2 = box_y.repeat_interleave(box_width_x).repeat_interleave(
        off_per_group[1:] - off_per_group[:-1])
    y = torch.arange(n_pix, device=device) - off_1 + off_2

    return x, y


def z_buffering(rect, x_proj, y_proj, dist, exact=False):
    """Z-buffering of rectangles with provided projection center and
    distance.

    :param rect:
    :param x_proj:
    :param y_proj:
    :param dist:
    :param exact:
    :return:
    """
    f = _z_buffering_cuda if rect.is_cuda else _z_buffering_cpu
    return f(rect, x_proj, y_proj, dist, exact)


def _z_buffering_cuda(rect, x_proj, y_proj, dist, exact):
    """CUDA implementation for z_buffering."""
    # Initialization
    device = x_proj.device
    n_points = rect.shape[0]
    assert n_points == x_proj.shape[0] == y_proj.shape[0] == dist.shape[0] > 0

    # Convert rectangles to flattened global pixel coordinates
    xy_max = rect.max()
    x_all_rect, y_all_rect = bbox_to_xy_grid_cuda(rect)
    xy_pix_all_rect = x_all_rect + y_all_rect * xy_max

    # Compute point values associated with all pixels of all rectangles
    areas = (rect[:, 1] - rect[:, 0]) * (rect[:, 3] - rect[:, 2])
    indices_all_rect = torch.arange(
        n_points, device=device).repeat_interleave(areas)
    depth_all_rect = dist.repeat_interleave(areas)

    # Compute which rect pixel coordinates are seen (ie their depth is
    # the smallest)
    _, seen_pix = torch_scatter.scatter_min(depth_all_rect, xy_pix_all_rect)
    seen_pix = seen_pix[xy_pix_all_rect.unique()]

    # Recover corresponding point indices and pixel coordinates
    indices = indices_all_rect[seen_pix]
    xy_pix_seen = xy_pix_all_rect[seen_pix]
    x_pix = xy_pix_seen % xy_max
    y_pix = xy_pix_seen // xy_max

    # When 'exact=True', we use the results from the previous projection
    # to extract the seen points. The output maps are sparse, as seen
    # points are only mapped to the center of their rectangles, without
    # artificially-built splatting masks.
    if exact:
        # Recover the local indices of seen points
        indices = indices.unique()

        # Recover corresponding rect-center coordinates
        x_pix = x_proj[indices].long()
        y_pix = y_proj[indices].long()

    return indices, x_pix, y_pix


@torch_to_numba
@njit(cache=True, nogil=True)
def _z_buffering_cpu(rect, x_proj, y_proj, dist, exact):
    """Numba implementation for z_buffering."""
    # Initialization
    n_points = rect.shape[0]
    assert n_points == x_proj.shape[0] == y_proj.shape[0] == dist.shape[0] > 0

    # Cropped depth map initialization
    d_max = dist.max() + 1
    buffer_size = (rect[:, 1].max() + 1, rect[:, 3].max() + 1)
    depth_map = np.full(buffer_size, d_max + 1, dtype=np.float32)

    # Cropped indices map initialization
    # We store indices in int64 so we assumes point indices are lower
    # than max int64 ~ 2.14 x 10^9.
    # We need the negative for empty pixels
    no_id = -1
    idx_map = np.full(buffer_size, no_id, dtype=np.int64)

    # Loop through indices for points in range and in FOV
    for i_point in range(dist.shape[0]):

        point_dist = dist[i_point]
        point_pix_mask = rect[i_point]

        # Update maps where point is closest recorded
        x_a, x_b, y_a, y_b = point_pix_mask
        for x in range(x_a, x_b):
            for y in range(y_a, y_b):
                if point_dist < depth_map[x, y]:
                    depth_map[x, y] = point_dist
                    # These indices can then be used to efficiently
                    # build the 'exact' maps without the need for
                    # 'np.isin', which is not supported un numba.
                    idx_map[x, y] = i_point

    # When 'exact=True', we use the results from the previous projection
    # to extract the seen points. The output maps are sparse, as seen
    # points are only mapped to the center of their rectangles, without
    # artificially-built splatting masks.
    if exact:
        # Recover the local indices of seen points
        idx_seen = np.unique(idx_map)
        idx_seen = idx_seen[idx_seen != no_id]

        # Reinitialize the output maps
        idx_map = np.full(buffer_size, no_id, dtype=np.int64)

        # Convert the pixel projection coordinates to int
        x_proj = x_proj.astype(np.int32)
        y_proj = y_proj.astype(np.int32)

        # Loop through the seen points and populate only the center of
        # rectangles in the maps. We can update maps without worrying
        # about occlusions here.
        if idx_seen.shape[0] > 0:
            for i_point in idx_seen:
                x = x_proj[i_point]
                y = y_proj[i_point]
                idx_map[x, y] = i_point

    # Recover final point indices and corresponding pixel coordinates
    x_pix, y_pix = np.where(idx_map != no_id)
    indices = np.zeros_like(x_pix)
    for i, (x, y) in enumerate(zip(x_pix, y_pix)):
        indices[i] = idx_map[x, y]

    return indices, x_pix, y_pix


# -------------------------------------------------------------------- #
#                     Visibility Method - Depth Map                    #
# -------------------------------------------------------------------- #

class DepthBasedVisibility(VisibilityModel, ABC):
    """Compute visibility model based on an input depth map. Points
    within a given distance threshold of the target depth are considered
    visible.
    """

    def __init__(
            self, crop_top=0, crop_bottom=0, r_max=30, r_min=0.5,
            depth_threshold=0.05):
        super().__init__(
            crop_top=crop_top, crop_bottom=crop_bottom, r_max=r_max,
            r_min=r_min)
        self.depth_threshold = depth_threshold

    def _run(
            self, x_proj, y_proj, dist, xyz, img_extrinsic, camera,
            depth_file=None, **kwargs):
        assert x_proj.shape[0] == y_proj.shape[0] == dist.shape[0] > 0

        # Read the depth map
        # TODO: only supports S3DIS-type depth map. Extend to other formats
        assert depth_file is not None, f'Please provide `depth_file`.'
        depth_map = read_s3dis_depth_map(
            depth_file, img_size=camera.size, empty=-1)
        depth_map = depth_map.to(x_proj.device)

        # Search point projections that are within depth_threshold of the
        # real depth
        dist_real = depth_map[x_proj.long(), y_proj.long()]
        indices = torch.where(
            (dist_real - dist).abs() <= self.depth_threshold)[0]

        return indices, x_proj[indices], y_proj[indices]


def read_s3dis_depth_map(path, img_size=None, empty=-1):
    """Read S3DIS-format depth map.

    Details from https://github.com/alexsax/2D-3D-Semantics
    "
    Depth images are stored as 16-bit PNGs and have a maximum depth of
    128m and a sensitivity of 1/512m. Missing values are encoded with
    the value 2^16 - 1. Note that [...] it [depth] is defined as the
    distance from the point-center of the camera in the
    (equirectangular) panoramic images.
    "

    :param path:
    :param img_size:
    :param empty:
    :return:
    """
    # Read depth map
    im = Image.open(path)
    if img_size is not None:
        im = im.resize(img_size, resample=Image.NEAREST)
    im = torch.from_numpy(np.array(im)).t()

    # Get mask of empty pixels
    empty_mask = im == 2 ** 16 - 1

    # Convert to meters and set empty pixels
    im = im / 512
    im[empty_mask] = empty

    return im


# -------------------------------------------------------------------- #
#                     Visibility Method - Biasutti                     #
# -------------------------------------------------------------------- #

class BiasuttiVisibility(VisibilityModel, ABC):
    """Compute visibility model based Biasutti et al. method as
    described in:

    "Visibility estimation in point clouds with variable density"
    Source: https://hal.archives-ouvertes.fr/hal-01812061/document
    """

    def __init__(
            self, crop_top=0, crop_bottom=0, r_max=30, r_min=0.5, k=75,
            margin=None, threshold=None):
        super().__init__(
            crop_top=crop_top, crop_bottom=crop_bottom, r_max=r_max,
            r_min=r_min)
        self.k = k
        self.margin = margin
        self.threshold = threshold

    def _run(self, x_proj, y_proj, dist, xyz, img_extrinsic, camera, **kwargs):
        assert x_proj.shape[0] == y_proj.shape[0] == dist.shape[0] > 0

        # Search k-nearest neighbors in the image pixel coordinate system
        neighbors = k_nn_image_system(
            x_proj, y_proj, k=self.k, x_margin=self.margin,
            x_width=camera.img_size[0])

        # Compute the visibility and recover visible point indices
        dist_nn = dist[neighbors]
        dist_min = dist_nn.min(dim=1).values
        dist_max = dist_nn.max(dim=1).values
        alpha = torch.exp(-((dist - dist_min) / (dist_max - dist_min)) ** 2)
        if self.threshold is None:
            threshold = alpha.mean()
        indices = torch.where(alpha >= threshold)[0]

        return indices, x_proj[indices], y_proj[indices]


def k_nn_image_system(
        x_proj, y_proj, k=75, x_margin=None, x_width=None):
    """Compute K-nearest neighbors in for points with coordinates
    (x_proj, y_proj) in the image coordinate system. If x_margin and
    x_width are provided, the image is wrapped along the X coordinates
    to search neighbors on the border. This is typically needed for
    spherical images.

    :param x_proj:
    :param y_proj:
    :param k:
    :param x_margin:
    :param x_width:
    :return:
    """
    assert x_margin is None or x_width > 0, \
        f'x_margin and x_width must both be provided for image wrapping.'

    # Prepare query and search sets for KNN. Optionally wrap image for
    # neighbor search by the border.
    xy_query = torch.stack((x_proj.float(), y_proj.float())).t()
    wrap_x = x_margin is not None and x_margin > 0 \
             and x_width is not None and x_width > 0
    if wrap_x:
        x_offset = torch.Tensor([[x_width, 0]]).float().to(xy_query.device)

        idx_left = torch.where(x_proj <= x_margin)[0]
        idx_right = torch.where(x_proj >= (x_width - x_margin))[0]

        xy_search_left = xy_query[idx_left] + x_offset
        xy_search_right = xy_query[idx_right] - x_offset

        xy_search = torch.cat((xy_query, xy_search_left, xy_search_right))
    else:
        xy_search = xy_query

    # K-NN search with sklearn
    # nn_search = NearestNeighbors(
    #     n_neighbors=k_nn, algorithm="kd_tree").fit(xy_search.cpu().numpy())
    # _, neighbors = nn_search.kneighbors(xy_query.cpu().numpy())
    # neighbors = torch.LongTensor(neighbors).to(xy_query.device)

    # K-NN search with KeOps
    xy_query = xy_query.contiguous()
    xy_search = xy_search.contiguous()
    xyz_query_keops = LazyTensor(xy_query[:, None, :])
    xyz_search_keops = LazyTensor(xy_search[None, :, :])
    d_keops = ((xyz_query_keops - xyz_search_keops) ** 2).sum(dim=2)
    neighbors = d_keops.argKmin(k, dim=1)
    del xyz_query_keops, xyz_search_keops, d_keops

    # Set the indices of margin points to their original values
    if wrap_x:
        n_points = x_proj.shape[0]
        n_points_left = idx_left.shape[0]

        is_left_margin = \
            (neighbors >= n_points) & (neighbors < n_points + n_points_left)
        neighbors[is_left_margin] = \
            idx_left[neighbors[is_left_margin] - n_points]

        is_right_margin = neighbors >= (n_points + n_points_left)
        neighbors[is_right_margin] = \
            idx_right[neighbors[is_right_margin] - n_points - n_points_left]

    return neighbors
