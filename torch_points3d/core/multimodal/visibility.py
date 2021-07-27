import numpy as np
import numba as nb
from numba import njit
import torch
import torch_scatter


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


# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------

@njit(cache=True, nogil=True, fastmath=True)
def field_of_view_cpu(x_pix, y_pix, y_min, y_max, mask=None):
    """

    :param x_pix:
    :param y_pix:
    :param y_min:
    :param y_max:
    :param mask:
    :return:
    """
    in_fov = np.logical_and(y_min <= y_pix, y_pix < y_max)
    if not mask is None:
        n_points = x_pix.shape[0]
        x_int = np.floor(x_pix).astype(np.uint32)
        y_int = np.floor(y_pix).astype(np.uint32)
        for i in range(n_points):
            if in_fov[i] and not mask[x_int[i], y_int[i]]:
                in_fov[i] = False
    return np.where(in_fov)[0]


def field_of_view_cuda(x_pix, y_pix, y_min, y_max, mask=None):
    """

    :param x_pix:
    :param y_pix:
    :param y_min:
    :param y_max:
    :param mask:
    :return:
    """
    in_fov = torch.logical_and(y_min <= y_pix, y_pix < y_max)
    if not mask is None:
        x_int = torch.floor(x_pix).long()
        y_int = torch.floor(y_pix).long()
        in_fov = torch.logical_and(in_fov, mask[x_int, y_int])
    return torch.where(in_fov)[0]


# ----------------------------------------------------------------------

@njit(cache=True, nogil=True)
def equirectangular_splat_cpu(
        x_proj, y_proj, dist, img_size=(1024, 512), crop_top=0,
        crop_bottom=0, voxel=0.03, k=0.2, d=10):
    """

    :param x_proj:
    :param y_proj:
    :param dist:
    :param img_size:
    :param crop_top:
    :param crop_bottom:
    :param voxel:
    :param k:
    :param d:
    :return:
    """
    # Compute angular width. 3D points' projected masks are grown based
    # on their distance. Close-by points are further grown with a
    # heuristic based on k and d.
    # Small angular widths assumption: tan(x)~x
    angular_width = (1 + k * np.exp(-dist / np.log(d))) * voxel / dist

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
        x_proj, y_proj, dist, img_size=(1024, 512), crop_top=0,
        crop_bottom=0, voxel=0.03, k=0.2, d=10):
    """

    :param x_proj:
    :param y_proj:
    :param dist:
    :param img_size:
    :param crop_top:
    :param crop_bottom:
    :param voxel:
    :param k:
    :param d:
    :return:
    """
    # Compute angular width. 3D points' projected masks are grown based
    # on their distance. Close-by points are further grown with a
    # heuristic based on k and d.
    # Small angular widths assumption: tan(x)~x
    angular_width = (1 + k * torch.exp(-dist / np.log(d))) * voxel / dist

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


# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------

def normalize_dist_cuda(dist, low=None, high=None):
    """

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


# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------

@torch_to_numba
@njit(cache=True, nogil=True)
def project_cpu(
        xyz_to_img, img_opk, img_mask=None, img_size=(1024, 512),
        crop_top=0, crop_bottom=0, r_max=30, r_min=0.5):
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
        x_proj, y_proj, crop_top, img_size[1] - crop_bottom, mask=img_mask)
    dist = dist[in_fov]
    indices = indices[in_fov]
    x_proj = x_proj[in_fov]
    y_proj = y_proj[in_fov]

    return indices, dist, x_proj, y_proj


def project_cuda(
        xyz_to_img, img_opk, img_mask=None, img_size=(1024, 512), crop_top=0,
        crop_bottom=0, r_max=30, r_min=0.5):
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
        x_proj, y_proj, crop_top, img_size[1] - crop_bottom, mask=img_mask)
    dist = dist[in_fov]
    indices = indices[in_fov]
    x_proj = x_proj[in_fov]
    y_proj = y_proj[in_fov]

    return indices, dist, x_proj, y_proj


# ----------------------------------------------------------------------

@torch_to_numba
@njit(cache=True, nogil=True)
def splatting_cpu(
        x_proj, y_proj, dist, img_size=(1024, 512), crop_top=0,
        crop_bottom=0, voxel=0.1, growth_k=0.2, growth_r=10, exact=False):
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
    :param growth_k:
    :param growth_r:
    :param exact:
    :return:
    """
    # Compute splatting masks for equirectangular images
    splat = equirectangular_splat_cpu(
        x_proj, y_proj, dist, img_size=img_size, crop_top=crop_top,
        crop_bottom=crop_bottom, voxel=voxel, k=growth_k, d=growth_r)

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


def splatting_cuda(
        x_proj, y_proj, dist, img_size=(1024, 512), crop_top=0,
        crop_bottom=0, voxel=0.1, growth_k=0.2, growth_r=10, exact=False):
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
    :param growth_k:
    :param growth_r:
    :param exact:
    :return:
    """
    # Initialization
    device = x_proj.device
    n_points = x_proj.shape[0]

    # Compute splatting masks for equirectangular images
    splat = equirectangular_splat_cuda(
        x_proj, y_proj, dist, img_size=img_size, crop_top=crop_top,
        crop_bottom=crop_bottom, voxel=voxel, k=growth_k, d=growth_r)

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


# ----------------------------------------------------------------------

def visibility(
        xyz_to_img, img_opk, img_mask=None, img_size=(1024, 512), crop_top=0,
        crop_bottom=0, r_max=30, r_min=0.5, method='splatting', use_cuda=False,
        voxel=0.1, growth_k=0.2, growth_r=10, exact=False, linearity=None,
        planarity=None, scattering=None, normals=None):
    """
    Compute the visibility of a point cloud with respect to a given
    camera pose.

    :param xyz_to_img:
    :param img_opk:
    :param img_mask:
    :param img_size:
    :param crop_top:
    :param crop_bottom:
    :param r_max:
    :param r_min:
    :param method:
    :param use_cuda:
    :param voxel:
    :param growth_k:
    :param growth_r:
    :param exact:
    :param linearity:
    :param planarity:
    :param scattering:
    :param normals:
    :return:
    """
    METHODS = ['splatting', 'depth_map', 'image_knn']
    assert method in METHODS, \
        f'Unknown method {method}, expected one of {METHODS}.'

    device = xyz_to_img.device
    if device.type == 'cuda':
        use_cuda = True
    elif not torch.cuda.is_available():
        use_cuda = False

    assert img_mask is None or img_mask.shape == img_size, \
        f'Expected img_mask to be a torch.BoolTensor of shape ' \
        f'img_size={img_size} but got size={img_mask.shape}.'

    if use_cuda:
        idx_1, dist, x_proj, y_proj = project_cuda(
            xyz_to_img, img_opk, img_mask=img_mask, img_size=img_size,
            crop_top=crop_top, crop_bottom=crop_bottom, r_max=r_max,
            r_min=r_min)
    else:
        idx_1, dist, x_proj, y_proj = project_cpu(
            xyz_to_img, img_opk, img_mask=img_mask, img_size=img_size,
            crop_top=crop_top, crop_bottom=crop_bottom, r_max=r_max,
            r_min=r_min)

    if method == 'splatting' and use_cuda:
        idx_2, x_pix, y_pix = splatting_cuda(
            x_proj, y_proj, dist, img_size=img_size, crop_top=crop_top,
            crop_bottom=crop_bottom, voxel=voxel, growth_k=growth_k,
            growth_r=growth_r, exact=exact)
    elif method == 'splatting' and not use_cuda:
        idx_2, x_pix, y_pix = splatting_cpu(
            x_proj, y_proj, dist, img_size=img_size, crop_top=crop_top,
            crop_bottom=crop_bottom, voxel=voxel, growth_k=growth_k,
            growth_r=growth_r, exact=exact)
    elif method == 'depth_map' and use_cuda:
        raise NotImplementedError
    elif method == 'depth_map' and not use_cuda:
        raise NotImplementedError
    elif method == 'image_knn' and use_cuda:
        raise NotImplementedError
    elif method == 'image_knn' and not use_cuda:
        raise NotImplementedError
    else:
        raise NotImplementedError

    # Keep data only for mapped point
    idx = idx_1[idx_2]
    x_proj = x_proj[idx_2]
    y_proj = y_proj[idx_2]
    dist = dist[idx_2]
    xyz_to_img = xyz_to_img[idx]

    # Compute mapping features
    has_mapping_features = linearity is not None and planarity is not None \
       and scattering is not None and normals is not None
    if has_mapping_features:
        linearity = linearity[idx]
        planarity = planarity[idx]
        scattering = scattering[idx]
        normals = normals[idx]

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

    out = {}
    out['idx'] = idx
    out['x'] = x_pix
    out['y'] = y_pix
    out['depth'] = dist
    if has_mapping_features:
        out['features'] = features

    return out
