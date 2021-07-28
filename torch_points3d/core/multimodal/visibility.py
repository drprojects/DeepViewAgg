import numpy as np
from numba import njit
import torch
import torch_scatter
from PIL import Image
from pykeops.torch import LazyTensor


def torch_to_numba(func):
    """Decorator intended for numba functions to be fed and return
    torch.Tensor arguments.

    :param func:
    :return:
    """

    def numbafy(x):
        return x.cpu().numpy() if isinstance(x, torch.Tensor) else x

    def torchify(x):
        return torch.from_numpy(x) if isinstance(x, np.ndarray) else x

    def wrapper_torch_to_numba(*args, **kwargs):
        args_numba = [numbafy(x) for x in args]
        kwargs_numba = {k: numbafy(v) for k, v in kwargs.items()}
        out = func(*args_numba, **kwargs_numba)
        if isinstance(out, list):
            out = [torchify(x) for x in out]
        elif isinstance(out, tuple):
            out = tuple([torchify(x) for x in list(out)])
        elif isinstance(out, dict):
            out = {k: torchify(v) for k, v in out.items()}
        else:
            out = torchify(out)
        return out

    return wrapper_torch_to_numba


# -------------------------------------------------------------------- #
#                           Camera Projection                          #
# -------------------------------------------------------------------- #

@njit(cache=True, nogil=True)
def pose_to_rotation_matrix_cpu(opk):
    """Compute the rotation matrix from an omega, phi kappa triplet on
    the CPU with numpy and numba.

    :param opk:
    :return:
    """
    # Omega, Phi, Kappa cos and sin
    co = np.cos(opk[0])
    so = np.sin(opk[0])
    cp = np.cos(opk[1])
    sp = np.sin(opk[1])
    ck = np.cos(opk[2])
    sk = np.sin(opk[2])

    # Omega, Phi, Kappa inverse rotation matrices
    M_o = np.array([[1.0, 0.0, 0.0],
                    [0.0, co, -so],
                    [0.0, so, co]], dtype=np.float32)

    M_p = np.array([[cp, 0.0, sp],
                    [0.0, 1.0, 0.0],
                    [-sp, 0.0, cp]], dtype=np.float32)

    M_k = np.array([[ck, -sk, 0.0],
                    [sk, ck, 0.0],
                    [0.0, 0.0, 1.0]], dtype=np.float32)

    # Global inverse rotation matrix to go from cartesian to
    # camera-system spherical coordinates
    M = np.dot(M_o, np.dot(M_p, M_k))

    return M


def pose_to_rotation_matrix_cuda(opk):
    """Compute the rotation matrix from an omega, phi kappa triplet on
    the GPU with torch and cuda.

    :param opk:
    :return:
    """
    # Omega, Phi, Kappa cos and sin
    co = torch.cos(opk[0])
    so = torch.sin(opk[0])
    cp = torch.cos(opk[1])
    sp = torch.sin(opk[1])
    ck = torch.cos(opk[2])
    sk = torch.sin(opk[2])

    # Omega, Phi, Kappa inverse rotation matrices
    device = opk.device
    M_o = torch.Tensor([[1.0, 0.0, 0.0],
                        [0.0, co, -so],
                        [0.0, so, co]]).float().to(device)

    M_p = torch.Tensor([[cp, 0.0, sp],
                        [0.0, 1.0, 0.0],
                        [-sp, 0.0, cp]]).float().to(device)

    M_k = torch.Tensor([[ck, -sk, 0.0],
                        [sk, ck, 0.0],
                        [0.0, 0.0, 1.0]]).float().to(device)

    # Global inverse rotation matrix to go from cartesian to
    # camera-system spherical coordinates
    M = torch.mm(M_o, torch.mm(M_p, M_k))

    return M


@njit(cache=True, nogil=True)
def norm_cpu(v):
    """Compute the L2 norm of row vectors of v on the CPU with numpy and
    numba.

    :param v:
    :return:
    """
    return np.sqrt((v ** 2).sum(axis=1))


def norm_cuda(v):
    """Compute the L2 norm of row vectors of v on the GPU with torch and
    cuda.

    :param v:
    :return:
    """
    return torch.linalg.norm(v, dim=1)


@njit(cache=True, nogil=True)
def equirectangular_projection_cpu(
        xyz_to_img, radius, img_rotation, img_size):
    """Compute the projection of 3D points into the image pixel coordinate
    system of an equirectangular camera on the GPU with numpy and numba.

    :param xyz_to_img:
    :param radius:
    :param img_rotation:
    :param img_size:
    :return:
    """
    # Convert point to camera coordinate system
    v = xyz_to_img.dot(img_rotation.transpose())

    # Equirectangular projection
    t = np.arctan2(v[:, 1], v[:, 0])
    p = np.arccos(v[:, 2] / radius)

    # Angles to pixel position
    width, height = img_size
    w_pix = ((width - 1) * (1 - t / np.pi) / 2) % width
    h_pix = ((height - 1) * p / np.pi) % height

    return w_pix, h_pix


def equirectangular_projection_cuda(
        xyz_to_img, radius, img_rotation, img_size):
    """Compute the projection of 3D points into the image pixel coordinate
    system of an equirectangular camera on the GPU with torch and cuda.

    :param xyz_to_img:
    :param radius:
    :param img_rotation:
    :param img_size:
    :return:
    """
    # Convert point to camera coordinate system
    v = xyz_to_img.mm(img_rotation.t())

    # Equirectangular projection
    t = torch.atan2(v[:, 1], v[:, 0])
    p = torch.acos(v[:, 2] / radius)

    # Angles to pixel position
    width, height = img_size
    w_pix = ((width - 1) * (1 - t / np.pi) / 2) % width
    h_pix = ((height - 1) * p / np.pi) % height

    return w_pix, h_pix


@njit(cache=True, nogil=True, fastmath=True)
def field_of_view_cpu(x_pix, y_pix, y_min, y_max, img_mask=None):
    """

    :param x_pix:
    :param y_pix:
    :param y_min:
    :param y_max:
    :param img_mask:
    :return:
    """
    in_fov = np.logical_and(y_min <= y_pix, y_pix < y_max)
    if not img_mask is None:
        n_points = x_pix.shape[0]
        x_int = np.floor(x_pix).astype(np.uint32)
        y_int = np.floor(y_pix).astype(np.uint32)
        for i in range(n_points):
            if in_fov[i] and not img_mask[x_int[i], y_int[i]]:
                in_fov[i] = False
    return np.where(in_fov)[0]


def field_of_view_cuda(x_pix, y_pix, y_min, y_max, img_mask=None):
    """

    :param x_pix:
    :param y_pix:
    :param y_min:
    :param y_max:
    :param img_mask:
    :return:
    """
    in_fov = torch.logical_and(y_min <= y_pix, y_pix < y_max)
    if not img_mask is None:
        x_int = torch.floor(x_pix).long()
        y_int = torch.floor(y_pix).long()
        in_fov = torch.logical_and(in_fov, img_mask[x_int, y_int])
    return torch.where(in_fov)[0]


@torch_to_numba
@njit(cache=True, nogil=True)
def project_cpu(
        xyz_to_img, img_opk, img_mask=None, img_size=(1024, 512), crop_top=0,
        crop_bottom=0, r_max=30, r_min=0.5, **kwargs):
    """

    :param xyz_to_img:
    :param img_opk:
    :param img_mask:
    :param img_size:
    :param crop_top:
    :param crop_bottom:
    :param r_max:
    :param r_min:
    :return:
    """
    assert img_mask is None or img_mask.shape == img_size, \
        f'Expected img_mask to be a torch.BoolTensor of shape ' \
        f'img_size={img_size} but got size={img_mask.shape}.'

    # We store indices in int64 format so we only accept indices up to
    # np.iinfo(np.int64).max
    num_points = xyz_to_img.shape[0]
    if num_points >= 9223372036854775807:
        raise OverflowError

    # Initialize the indices to keep track of selected points
    indices = np.arange(num_points)

    # Rotation matrix from image Euler angle pose
    img_rotation = pose_to_rotation_matrix_cpu(img_opk)

    # Remove points outside of image range
    dist = norm_cpu(xyz_to_img)
    in_range = np.where(np.logical_and(r_min < dist, dist < r_max))[0]
    xyz_to_img = xyz_to_img[in_range]
    dist = dist[in_range]
    indices = indices[in_range]

    # Project points to float pixel coordinates
    x_proj, y_proj = equirectangular_projection_cpu(
        xyz_to_img, dist, img_rotation, img_size)

    # Remove points outside of camera field of view
    in_fov = field_of_view_cpu(
        x_proj, y_proj, crop_top, img_size[1] - crop_bottom, img_mask=img_mask)
    dist = dist[in_fov]
    indices = indices[in_fov]
    x_proj = x_proj[in_fov]
    y_proj = y_proj[in_fov]

    return indices, dist, x_proj, y_proj


def project_cuda(
        xyz_to_img, img_opk, img_mask=None, img_size=(1024, 512), crop_top=0,
        crop_bottom=0, r_max=30, r_min=0.5, **kwargs):
    """

    :param xyz_to_img:
    :param img_opk:
    :param img_mask:
    :param img_size:
    :param crop_top:
    :param crop_bottom:
    :param r_max:
    :param r_min:
    :return:
    """
    assert img_mask is None or img_mask.shape == img_size, \
        f'Expected img_mask to be a torch.BoolTensor of shape ' \
        f'img_size={img_size} but got size={img_mask.shape}.'

    # We store indices in int64 format so we only accept indices up to
    # torch.iinfo(torch.long).max
    num_points = xyz_to_img.shape[0]
    if num_points >= 9223372036854775807:
        raise OverflowError

    # Initialize the indices to keep track of selected points
    indices = torch.arange(num_points, device=xyz_to_img.device)

    # Rotation matrix from image Euler angle pose
    img_rotation = pose_to_rotation_matrix_cuda(img_opk)

    # Remove points outside of image range
    dist = norm_cuda(xyz_to_img)
    in_range = torch.where(torch.logical_and(r_min < dist, dist < r_max))[0]
    xyz_to_img = xyz_to_img[in_range]
    dist = dist[in_range]
    indices = indices[in_range]

    # Project points to float pixel coordinates
    x_proj, y_proj = equirectangular_projection_cuda(
        xyz_to_img, dist, img_rotation, img_size)

    # Remove points outside of camera field of view
    in_fov = field_of_view_cuda(
        x_proj, y_proj, crop_top, img_size[1] - crop_bottom, img_mask=img_mask)
    dist = dist[in_fov]
    indices = indices[in_fov]
    x_proj = x_proj[in_fov]
    y_proj = y_proj[in_fov]

    return indices, dist, x_proj, y_proj


# -------------------------------------------------------------------- #
#                     Visibility Method - Splatting                    #
# -------------------------------------------------------------------- #

@njit(cache=True, nogil=True)
def equirectangular_splat_cpu(
        x_proj, y_proj, dist, img_size=(1024, 512), crop_top=0, crop_bottom=0,
        voxel=0.03, k_swell=0.2, d_swell=10):
    """

    :param x_proj:
    :param y_proj:
    :param dist:
    :param img_size:
    :param crop_top:
    :param crop_bottom:
    :param voxel:
    :param k_swell:
    :param d_swell:
    :return:
    """
    # Compute angular width. 3D points' projected masks are grown based
    # on their distance. Close-by points are further swollen with a
    # heuristic based on k_swell and d_swell.
    # Small angular widths assumption: tan(x)~x
    angular_width = \
        (1 + k_swell * np.exp(-dist / np.log(d_swell))) * voxel / dist

    # Compute Y angular width
    # NB: constant for equirectangular projection
    angular_res_y = angular_width * img_size[1] / np.pi

    # Compute X angular width
    # NB: function of latitude for equirectangular projection
    a = angular_width * img_size[0] / (2.0 * np.pi)
    b = np.pi / img_size[1]
    angular_res_x = a / (np.sin(b * y_proj) + 0.001)

    # NB: stack+transpose faster than column stack
    splat_xy_width = np.stack((angular_res_x, angular_res_y)).transpose()

    # Compute projection masks bounding box pixel coordinates
    x_a = np.empty_like(x_proj, dtype=np.float32)
    x_b = np.empty_like(x_proj, dtype=np.float32)
    y_a = np.empty_like(x_proj, dtype=np.float32)
    y_b = np.empty_like(x_proj, dtype=np.float32)
    np.round(x_proj - splat_xy_width[:, 0] / 2, 0, x_a)
    np.round(x_proj + splat_xy_width[:, 0] / 2 + 1, 0, x_b)
    np.round(y_proj - splat_xy_width[:, 1] / 2, 0, y_a)
    np.round(y_proj + splat_xy_width[:, 1] / 2 + 1, 0, y_b)
    splat = np.stack((x_a, x_b, y_a, y_b)).transpose().astype(np.int32)

    # Adjust masks at the image border
    x_min = 0
    x_max = img_size[0]
    y_min = crop_top
    y_max = img_size[1] - crop_bottom
    for i in range(splat.shape[0]):
        if splat[i, 0] < x_min:
            splat[i, 0] = x_min
        if splat[i, 0] > x_max - 1:
            splat[i, 0] = x_max - 1

        if splat[i, 1] < x_min + 1:
            splat[i, 1] = x_min + 1
        if splat[i, 1] > x_max:
            splat[i, 1] = x_max

        if splat[i, 2] < y_min:
            splat[i, 2] = y_min
        if splat[i, 2] > y_max - 1:
            splat[i, 2] = y_max - 1

        if splat[i, 3] < y_min + 1:
            splat[i, 3] = y_min + 1
        if splat[i, 3] > y_max:
            splat[i, 3] = y_max

    return splat


def equirectangular_splat_cuda(
        x_proj, y_proj, dist, img_size=(1024, 512), crop_top=0, crop_bottom=0,
        voxel=0.03, k_swell=0.2, d_swell=10):
    """

    :param x_proj:
    :param y_proj:
    :param dist:
    :param img_size:
    :param crop_top:
    :param crop_bottom:
    :param voxel:
    :param k_swell:
    :param d_swell:
    :return:
    """
    # Compute angular width. 3D points' projected masks are grown based
    # on their distance. Close-by points are further swollen with a
    # heuristic based on k_swell and d_swell.
    # Small angular widths assumption: tan(x)~x
    angular_width = \
        (1 + k_swell * torch.exp(-dist / np.log(d_swell))) * voxel / dist

    # Compute Y angular width
    # NB: constant for equirectangular projection
    angular_res_y = angular_width * img_size[1] / np.pi

    # Compute X angular width
    # NB: function of latitude for equirectangular projection
    a = angular_width * img_size[0] / (2.0 * np.pi)
    b = np.pi / img_size[1]
    angular_res_x = a / (torch.sin(b * y_proj) + 0.001)
    splat_xy_width = torch.stack((angular_res_x, angular_res_y)).t()

    # Compute projection masks bounding box pixel coordinates
    x_a = torch.round(x_proj - splat_xy_width[:, 0] / 2)
    x_b = torch.round(x_proj + splat_xy_width[:, 0] / 2 + 1)
    y_a = torch.round(y_proj - splat_xy_width[:, 1] / 2)
    y_b = torch.round(y_proj + splat_xy_width[:, 1] / 2 + 1)
    splat = torch.stack((x_a, x_b, y_a, y_b)).t().long()

    # Adjust masks at the image border
    x_min = 0
    x_max = img_size[0]
    y_min = crop_top
    y_max = img_size[1] - crop_bottom
    splat[:, 0] = splat[:, 0].clamp(min=x_min, max=x_max - 1)
    splat[:, 1] = splat[:, 1].clamp(min=x_min + 1, max=x_max)
    splat[:, 2] = splat[:, 2].clamp(min=y_min, max=y_max - 1)
    splat[:, 3] = splat[:, 3].clamp(min=y_min + 1, max=y_max)

    return splat


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
    assert bbox.dim() == 2 and bbox.shape[1] == 4
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


@torch_to_numba
@njit(cache=True, nogil=True)
def visibility_from_splatting_cpu(
        x_proj, y_proj, dist, img_size=(1024, 512), crop_top=0, crop_bottom=0,
        voxel=0.1, k_swell=0.2, d_swell=10, exact=False, **kwargs):
    """Compute visibility model with splatting on the CPU with numpy and
    numba.

    Although top and bottom cropping can be specified, the returned
    coordinates are expressed in the non-cropped image pixel coordinate
    system.

    :param x_proj:
    :param y_proj:
    :param dist:
    :param img_size:
    :param crop_top:
    :param crop_bottom:
    :param voxel:
    :param k_swell:
    :param d_swell:
    :param exact:
    :return:
    """
    # Compute splatting masks for equirectangular images
    splat = equirectangular_splat_cpu(
        x_proj, y_proj, dist, img_size=img_size, crop_top=crop_top,
        crop_bottom=crop_bottom, voxel=voxel, k_swell=k_swell, d_swell=d_swell)

    # Cropped depth map initialization
    d_max = dist.max() + 1
    cropped_img_size = (img_size[0], img_size[1] - crop_bottom - crop_top)
    depth_map = np.full(cropped_img_size, d_max + 1, dtype=np.float32)
    splat[:, 2:] -= crop_top

    # Cropped indices map initialization
    # We store indices in int64 so we assumes point indices are lower
    # than max int64 ~ 2.14 x 10^9.
    # We need the negative for empty pixels
    no_id = -1
    idx_map = np.full(cropped_img_size, no_id, dtype=np.int64)

    # Loop through indices for points in range and in FOV
    for i_point in range(dist.shape[0]):

        point_dist = dist[i_point]
        point_pix_mask = splat[i_point]

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
    # points are only mapped to the center of their splats, without
    # artificially-built splatting masks.
    if exact:
        # Recover the local indices of seen points
        idx_seen = np.unique(idx_map)
        idx_seen = idx_seen[idx_seen != no_id]

        # Reinitialize the output maps
        idx_map = np.full(cropped_img_size, no_id, dtype=np.int64)

        # Convert the pixel projection coordinates to int
        x_proj = x_proj.astype(np.int32)
        y_proj = y_proj.astype(np.int32)

        # Loop through the seen points and populate only the center of
        # splats in the maps. We can update maps without worrying about
        # occlusions here.
        if idx_seen.shape[0] > 0:
            for i_point in idx_seen:
                x = x_proj[i_point]
                y = y_proj[i_point] - crop_top
                idx_map[x, y] = i_point

    # Recover final point indices and corresponding pixel coordinates
    x_pix, y_pix = np.where(idx_map != no_id)
    indices = np.zeros_like(x_pix)
    for i, (x, y) in enumerate(zip(x_pix, y_pix)):
        indices[i] = idx_map[x, y]

    return indices, x_pix, y_pix + crop_top


def visibility_from_splatting_cuda(
        x_proj, y_proj, dist, img_size=(1024, 512), crop_top=0, crop_bottom=0,
        voxel=0.1, k_swell=0.2, d_swell=10, exact=False, **kwargs):
    """Compute visibility model with splatting on the GPU with torch and
    cuda.

    Although top and bottom cropping can be specified, the returned
    coordinates are expressed in the non-cropped image pixel coordinate
    system.

    :param x_proj:
    :param y_proj:
    :param dist:
    :param img_size:
    :param crop_top:
    :param crop_bottom:
    :param voxel:
    :param k_swell:
    :param d_swell:
    :param exact:
    :return:
    """
    # Initialization
    device = x_proj.device
    n_points = x_proj.shape[0]

    # Compute splatting masks for equirectangular images
    splat = equirectangular_splat_cuda(
        x_proj, y_proj, dist, img_size=img_size, crop_top=crop_top,
        crop_bottom=crop_bottom, voxel=voxel, k_swell=k_swell, d_swell=d_swell)

    # Convert splats to flattened global pixel coordinates
    x_all_splat, y_all_splat = bbox_to_xy_grid_cuda(splat)
    xy_pix_all_splat = x_all_splat + y_all_splat * max(img_size)

    # Compute point values associated with all pixels of all splats
    areas = (splat[:, 1] - splat[:, 0]) * (splat[:, 3] - splat[:, 2])
    indices_all_splats = torch.arange(
        n_points, device=device).repeat_interleave(areas)
    depth_all_splat = dist.repeat_interleave(areas)

    # Compute which splat pixel coordinates are seen (ie their depth is
    # the smallest)
    _, seen_pix = torch_scatter.scatter_min(depth_all_splat, xy_pix_all_splat)
    seen_pix = seen_pix[xy_pix_all_splat.unique()]

    # Recover corresponding point indices and pixel coordinates
    indices = indices_all_splats[seen_pix]
    xy_pix_seen = xy_pix_all_splat[seen_pix]
    x_pix = xy_pix_seen % max(img_size)
    y_pix = xy_pix_seen // max(img_size)

    # When 'exact=True', we use the results from the previous projection
    # to extract the seen points. The output maps are sparse, as seen
    # points are only mapped to the center of their splats, without
    # artificially-built splatting masks.
    if exact:
        # Recover the local indices of seen points
        indices = indices.unique()

        # Recover corresponding splat-center coordinates
        x_pix = x_proj[indices].long()
        y_pix = y_proj[indices].long()

    return indices, x_pix, y_pix


# -------------------------------------------------------------------- #
#                     Visibility Method - Depth Map                    #
# -------------------------------------------------------------------- #

def read_s3dis_depth_map(path, img_size=None, empty=-1):
    """Read S3DIS-format depth map.

    Details from https://github.com/alexsax/2D-3D-Semantics
    "
    Depth images are stored as 16-bit PNGs and have a maximum depth of
    128m and a sensitivity of 1/512m. Missing values are encoded with
    the value 2^16 - 1. Note that [...] it [depth] is defined as the
    distance from the point-center of the camera in the
    [equirectangular] panoramics.
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


def visibility_from_depth_map(
        x_proj, y_proj, dist, depth_map_path=None, img_size=(1024, 512),
        depth_threshold=0.05, **kwargs):
    """Compute visibility model based on an input depth map. Points
    within a given threshold of the target depth are considered visible.

    :param x_proj:
    :param y_proj:
    :param dist:
    :param depth_map_path:
    :param img_size:
    :param depth_threshold:
    :return:
    """
    # Read the depth map
    assert depth_map_path is not None, f'Please provide depth_map_path.'
    depth_map = read_s3dis_depth_map(depth_map_path, img_size=img_size, empty=-1)
    depth_map = depth_map.to(x_proj.device)

    # Search point projections that are within depth_threshold of the
    # real depth
    dist_real = depth_map[x_proj.long(), y_proj.long()]
    indices = torch.where((dist_real - dist).abs() <= depth_threshold)[0]

    return indices, x_proj[indices], y_proj[indices]


# -------------------------------------------------------------------- #
#                     Visibility Method - Biasutti                     #
# -------------------------------------------------------------------- #

def k_nn_image_system(
        x_proj, y_proj, k=30, x_margin=None, x_width=None):
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


def visibility_biasutti(
        x_proj, y_proj, dist, img_size=None, biasutti_k=75,
        biasutti_margin=None, biasutti_threshold=None, **kwargs):
    """Compute visibility model based Biasutti et al. method as
    described in:

    "Visibility estimation in point clouds with variable density"
    Source: https://hal.archives-ouvertes.fr/hal-01812061/document

    :param x_proj:
    :param y_proj:
    :param dist:
    :param img_size:
    :param biasutti_k:
    :param biasutti_margin:
    :param biasutti_threshold:
    :return:
    """
    # Search k-nearest neighbors in the image pixel coordinate system
    neighbors = k_nn_image_system(
        x_proj, y_proj, k=biasutti_k, x_margin=biasutti_margin,
        x_width=img_size[0])

    # Compute the visibility and recover visible point indices
    dist_nn = dist[neighbors]
    dist_min = dist_nn.min(dim=1).values
    dist_max = dist_nn.max(dim=1).values
    alpha = torch.exp(-((dist - dist_min) / (dist_max - dist_min))**2)
    if biasutti_threshold is None:
        biasutti_threshold = alpha.mean()
    indices = torch.where(alpha >= biasutti_threshold)[0]

    return indices, x_proj[indices], y_proj[indices]


# -------------------------------------------------------------------- #
#                           Mapping Features                           #
# -------------------------------------------------------------------- #

def normalize_dist_cuda(dist, low=None, high=None):
    """Rescale distances to [0, 1].

    :param dist:
    :param low:
    :param high:
    :return:
    """
    d_min = low
    d_max = high
    dist = dist.float()
    if low is None:
        d_min = dist.min()
    if high is None:
        d_max = dist.max()
    return ((dist - d_min) / (d_max + 1e-4)).float()


def orientation_cuda(u, v, requires_scaling=False):
    """Orientation is defined as |cos(theta)| with theta the angle
    between the u and v. By default, u and v are assumed to be already
    unit-scaled, use 'requires_scaling' if that is not the case.

    :param u:
    :param v:
    :param requires_scaling:
    :return:
    """
    orientation = torch.zeros(u.shape[0], device=u.device, dtype=torch.float)
    u = u.float()
    v = v.float()

    if v is None:
        return orientation

    if requires_scaling:
        u = u / (norm_cuda(u) + 1e-4).reshape((-1, 1)).float()
        v = v / (norm_cuda(v) + 1e-4).reshape((-1, 1)).float()

    orientation = (u * v).abs().sum(dim=1)
    # orientation[torch.where(orientation > 1)] = 0

    return orientation


def postprocess_features(
        xyz_to_img, y_proj, dist, linearity, planarity, scattering, normals,
        img_size=(1024, 512), r_max=30, r_min=0.5, **kwargs):

    # Compute the N x F array of pointwise projection features carrying:
    #     - normalized depth
    #     - linearity
    #     - planarity
    #     - scattering
    #     - orientation to the surface
    #     - normalized pixel height
    depth = normalize_dist_cuda(dist, low=r_min, high=r_max)
    orientation = orientation_cuda(
        xyz_to_img / (dist + 1e-4).reshape((-1, 1)), normals)
    height = (y_proj / img_size[1]).float()
    features = torch.stack(
        (depth, linearity, planarity, scattering, orientation, height)).t()

    return features


# -------------------------------------------------------------------- #
#                              Visibility                              #
# -------------------------------------------------------------------- #

def visibility(
        xyz_to_img, img_opk, method='splatting', linearity=None, planarity=None,
        scattering=None, normals=None, use_cuda=True, **kwargs):
    """Compute the visibility of a point cloud with respect to a given
    camera pose.

    :param xyz_to_img:
    :param img_opk:
    :param method:
    :param linearity:
    :param planarity:
    :param scattering:
    :param normals:
    :param use_cuda:
    :return:
    """
    METHODS = ['splatting', 'depth_map', 'biasutti']
    assert method in METHODS, \
        f'Unknown method {method}, expected one of {METHODS}.'

    if xyz_to_img.device.type == 'cuda':
        use_cuda = True
    elif not torch.cuda.is_available():
        use_cuda = False

    if use_cuda:
        idx_1, dist, x_proj, y_proj = project_cuda(
            xyz_to_img, img_opk, **kwargs)
    else:
        idx_1, dist, x_proj, y_proj = project_cpu(xyz_to_img, img_opk, **kwargs)

    if method == 'splatting' and use_cuda:
        idx_2, x_pix, y_pix = visibility_from_splatting_cuda(
            x_proj, y_proj, dist, **kwargs)
    elif method == 'splatting' and not use_cuda:
        idx_2, x_pix, y_pix = visibility_from_splatting_cpu(
            x_proj, y_proj, dist, **kwargs)
    elif method == 'depth_map':
        idx_2, x_pix, y_pix = visibility_from_depth_map(
            x_proj, y_proj, dist, **kwargs)
    elif method == 'biasutti':
        idx_2, x_pix, y_pix = visibility_biasutti(
            x_proj, y_proj, dist, **kwargs)
    else:
        raise NotImplementedError

    # Keep data only for mapped point
    idx = idx_1[idx_2]
    dist = dist[idx_2]

    out = {}
    out['idx'] = idx
    out['x'] = x_pix
    out['y'] = y_pix
    out['depth'] = dist

    # Compute mapping features
    if linearity is not None and planarity is not None \
            and scattering is not None and normals is not None:
        out['features'] = postprocess_features(
            xyz_to_img[idx], y_proj[idx_2], dist, linearity[idx],
            planarity[idx], scattering[idx], normals[idx], **kwargs)

    return out

# TODO: support other camera models than equirectangular image
# TODO: support other depth map files formats. For now, only S3DIS format supported
