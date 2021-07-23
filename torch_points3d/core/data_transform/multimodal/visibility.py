import numpy as np
import numba as nb
from numba import njit
import torch
import torch_scatter


def torch_to_numba(func):
    """Decorator intended for numba functions to be fed and return
    torch.Tensor arguments.
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
    """
    return np.sqrt((v ** 2).sum(axis=1))


def norm_cuda(v):
    """Compute the L2 norm of row vectors of v on the GPU with torch and
    cuda.
    """
    return torch.linalg.norm(v, dim=1)


# ----------------------------------------------------------------------

@njit(cache=True, nogil=True)
def equirectangular_projection_cpu(
        xyz_to_img, radius, img_rotation, img_size):
    """Compute the projection of 3D points into the image pixel coordinate
    system of an equirectangular camera on the GPU with numpy and numba.
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
def field_of_view_cpu(x_pix, y_pix, crop_top, crop_bottom, mask=None):
    in_fov = np.logical_and(crop_top <= y_pix, y_pix < crop_bottom)
    if not mask is None:
        n_points = x_pix.shape[0]
        x_int = np.floor(x_pix).astype(np.uint32)
        y_int = np.floor(y_pix).astype(np.uint32)
        for i in range(n_points):
            if in_fov[i] and not mask[x_int[i], y_int[i]]:
                in_fov[i] = False
    return np.where(in_fov)[0]


def field_of_view_cuda(x_pix, y_pix, crop_top, crop_bottom, mask=None):
    in_fov = torch.logical_and(crop_top <= y_pix, y_pix < crop_bottom)
    if not mask is None:
        x_int = torch.floor(x_pix).long()
        y_int = torch.floor(y_pix).long()
        in_fov = torch.logical_and(in_fov, mask[x_int, y_int])
    return torch.where(in_fov)[0]


# ----------------------------------------------------------------------

# @njit(cache=True, nogil=True)
# def array_pixel_width_cpu(
#         y_pix, dist, img_size=(1024, 512), voxel=0.03, k=0.2, d=10):
#     # Compute angular width
#     # Pixel are grown based on their dist
#     # Small angular widths assumption: tan(x)~x
#     # Close-by points are further grown with a heuristic based on k and d
#     angular_width = (1 + k * np.exp(-dist / np.log(d))) * voxel / dist
#
#     # Compute Y angular width
#     # NB: constant for equirectangular projection
#     angular_res_y = angular_width * img_size[1] / np.pi
#
#     # Compute X angular width
#     # NB: function of latitude for equirectangular projection
#     a = angular_width * img_size[0] / (2.0 * np.pi)
#     b = np.pi / img_size[1]
#     angular_res_x = a / (np.sin(b * y_pix) + 0.001)
#
#     # NB: stack+transpose faster than column stack
#     return np.stack((angular_res_x, angular_res_y)).transpose()


# ----------------------------------------------------------------------

# @njit(cache=True, nogil=True)
# def pixel_masks_cpu(x_pix, y_pix, width_pix):
#     x_a = np.empty_like(x_pix, dtype=np.float32)
#     x_b = np.empty_like(x_pix, dtype=np.float32)
#     y_a = np.empty_like(x_pix, dtype=np.float32)
#     y_b = np.empty_like(x_pix, dtype=np.float32)
#     np.round(x_pix - width_pix[:, 0] / 2, 0, x_a)
#     np.round(x_pix + width_pix[:, 0] / 2 + 1, 0, x_b)
#     np.round(y_pix - width_pix[:, 1] / 2, 0, y_a)
#     np.round(y_pix + width_pix[:, 1] / 2 + 1, 0, y_b)
#     return np.stack((x_a, x_b, y_a, y_b)).transpose().astype(np.int32)


# ----------------------------------------------------------------------

# @njit(cache=True, nogil=True)
# def border_pixel_masks_cpu(pix_masks, x_min, x_max, y_min, y_max):
#     for i in range(pix_masks.shape[0]):
#         if pix_masks[i, 0] < x_min:
#             pix_masks[i, 0] = x_min
#         if pix_masks[i, 1] > x_max:
#             pix_masks[i, 1] = x_max
#         if pix_masks[i, 2] < y_min:
#             pix_masks[i, 2] = y_min
#         if pix_masks[i, 3] > y_max:
#             pix_masks[i, 3] = y_max
#     return pix_masks

@njit(cache=True, nogil=True)
def equirectangular_splat_cpu(
        x_proj, y_proj, dist, img_size=(1024, 512), crop_top=0,
        crop_bottom=0, voxel=0.03, k=0.2, d=10):
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
        if splat[i, 1] > x_max:
            splat[i, 1] = x_max
        if splat[i, 2] < y_min:
            splat[i, 2] = y_min
        if splat[i, 3] > y_max:
            splat[i, 3] = y_max

    # Remove y-crop offset
    splat[:, 2:] -= crop_top

    return splat


def equirectangular_splat_cuda(
        x_proj, y_proj, dist, img_size=(1024, 512), crop_top=0,
        crop_bottom=0, voxel=0.03, k=0.2, d=10, offset=False):
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
    splat[:, 0] = splat[:, 0].clamp(min=0, max=img_size[0] - 1)
    splat[:, 1] = splat[:, 1].clamp(min=0, max=img_size[0])
    splat[:, 2] = splat[:, 2].clamp(min=crop_top, max=img_size[1] - crop_bottom - 1)
    splat[:, 3] = splat[:, 3].clamp(min=crop_top, max=img_size[1] - crop_bottom)

    print()
    mask = splat[:, 2] >= splat[:, 3]
    print(splat[mask, 2])
    print(splat[mask, 2])
    print()

    # Remove y-crop offset
    if offset:
        splat[:, 2:] -= crop_top

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

    print(box_width_y[box_width_y <= 0])

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

@njit(cache=True, nogil=True)
def normalize_dist_cpu(dist, low=None, high=None):
    d_min = low
    d_max = high
    dist = dist.astype(np.float32)
    if low is None:
        d_min = dist.min()
    if high is None:
        d_max = dist.max()
    return ((dist - d_min) / (d_max + 1e-4)).astype(np.float32)


# ----------------------------------------------------------------------

@njit(cache=True, nogil=True)
def orientation_cpu(u, v, requires_scaling=False):
    """Orientation is defined as |cos(theta)| with theta the angle
    between the u and v. By default, u and v are assumed to be already
    unit-scaled, use 'requires_scaling' if that is not the case.
    """
    orientation = np.zeros(u.shape[0], dtype=np.float32)
    u = u.astype(np.float32)
    v = v.astype(np.float32)

    if v is None:
        return orientation

    if requires_scaling:
        u = u / (norm_cpu(u) + 1e-4).reshape((-1, 1)).astype(np.float32)
        v = v / (norm_cpu(v) + 1e-4).reshape((-1, 1)).astype(np.float32)

    orientation = np.abs((u * v).sum(axis=1))
    # orientation[np.where(orientation > 1)] = 0

    return orientation


# ----------------------------------------------------------------------

@njit(cache=True, nogil=True)
def normalize_height_cpu(pixel_height, height):
    return (pixel_height / height).astype(np.float32)


# ----------------------------------------------------------------------

@njit(cache=True, nogil=True)
def splatting_depth(
        xyz_to_img, img_opk, img_mask=None, img_size=(1024, 512), crop_top=0,
        crop_bottom=0, voxel=0.1, r_max=30, r_min=0.5, growth_k=0.2,
        growth_r=10, empty=0):
    # Rotation matrix from image Euler angle pose
    img_rotation = pose_to_rotation_matrix_cpu(img_opk)

    # Remove points outside of image range
    dist = norm_cpu(xyz_to_img)
    in_range = np.where(np.logical_and(r_min < dist, dist < r_max))[0]

    # Project points to float pixel coordinates
    x_proj, y_proj = equirectangular_projection_cpu(
        xyz_to_img[in_range], dist[in_range], img_rotation, img_size)

    # Remove points outside of camera field of view
    in_fov = field_of_view_cpu(
        x_proj, y_proj, crop_top, img_size[1] - crop_bottom, mask=img_mask)

    # Compute projection pixel patches sizes 
    width_pix = array_pixel_width_cpu(
        y_proj[in_fov], dist[in_range][in_fov], img_size=img_size,
        voxel=voxel, k=growth_k, d=growth_r)
    pix_masks = pixel_masks_cpu(x_proj[in_fov], y_proj[in_fov], width_pix)
    pix_masks = border_pixel_masks_cpu(
        pix_masks, 0, img_size[0], crop_top, img_size[1] - crop_bottom)
    pix_masks[:, 2:] -= crop_top  # Remove y-crop offset

    # Cropped maps initialization
    cropped_img_size = (img_size[0], img_size[1] - crop_bottom - crop_top)
    depth_map = np.full(cropped_img_size, r_max + 1, np.float32)
    # undistort = np.sin(np.pi * np.arange(
    #     crop_top, img_size[1] - crop_bottom) / img_size[1]) + 0.001

    # Loop through indices for points in range and in FOV
    dist = dist[in_range][in_fov]
    for i_point in range(dist.shape[0]):

        point_dist = dist[i_point]
        point_pix_mask = pix_masks[i_point]

        # Update maps where point is closest recorded
        x_a, x_b, y_a, y_b = point_pix_mask
        for x in range(x_a, x_b):
            for y in range(y_a, y_b):
                if point_dist < depth_map[x, y]:
                    depth_map[x, y] = point_dist

    # Set empty pixels to default empty value
    for x in range(depth_map.shape[0]):
        for y in range(depth_map.shape[1]):
            if depth_map[x, y] > r_max:
                depth_map[x, y] = empty

    # Restore the cropped areas
    cropped_map_top = np.full((img_size[0], crop_top), empty, np.float32)
    cropped_map_bottom = np.full((img_size[0], crop_bottom), empty, np.float32)
    depth_map = np.concatenate(
        (cropped_map_top, depth_map, cropped_map_bottom), axis=1)

    return depth_map


# ----------------------------------------------------------------------

@njit(cache=True, nogil=True)
def splatting_rgb(
        xyz_to_img, rgb, img_opk, img_mask=None, img_size=(1024, 512),
        crop_top=0, crop_bottom=0, voxel=0.1, r_max=30, r_min=0.5, growth_k=0.2,
        growth_r=10, empty=0):
    # Rotation matrix from image Euler angle pose
    img_rotation = pose_to_rotation_matrix_cpu(img_opk)

    # Remove points outside of image range
    dist = norm_cpu(xyz_to_img)
    in_range = np.where(np.logical_and(r_min < dist, dist < r_max))[0]

    # Project points to float pixel coordinates
    x_proj, y_proj = equirectangular_projection_cpu(
        xyz_to_img[in_range], dist[in_range], img_rotation, img_size)

    # Remove points outside of camera field of view
    in_fov = field_of_view_cpu(
        x_proj, y_proj, crop_top, img_size[1] - crop_bottom, mask=img_mask)

    # Compute projection pixel patches sizes 
    width_pix = array_pixel_width_cpu(
        y_proj[in_fov], dist[in_range][in_fov], img_size=img_size,
        voxel=voxel, k=growth_k, d=growth_r)
    pix_masks = pixel_masks_cpu(x_proj[in_fov], y_proj[in_fov], width_pix)
    pix_masks = border_pixel_masks_cpu(
        pix_masks, 0, img_size[0], crop_top, img_size[1] - crop_bottom)
    pix_masks[:, 2:] -= crop_top  # Remove y-crop offset

    # Cropped maps initialization
    cropped_img_size = (img_size[0], img_size[1] - crop_bottom - crop_top)
    depth_map = np.full(cropped_img_size, r_max + 1, np.float32)
    rgb_map = np.zeros((*cropped_img_size, 3), dtype=np.int16)
    # undistort = np.sin(np.pi * np.arange(
    #     crop_top, img_size[1] - crop_bottom) / img_size[1]) + 0.001

    # Loop through indices for points in range and in FOV
    dist = dist[in_range][in_fov]
    rgb = rgb[in_range][in_fov]
    for i_point in range(dist.shape[0]):

        point_dist = dist[i_point]
        point_rgb = rgb[i_point]
        point_pix_mask = pix_masks[i_point]

        # Update maps where point is closest recorded
        x_a, x_b, y_a, y_b = point_pix_mask
        for x in range(x_a, x_b):
            for y in range(y_a, y_b):
                if point_dist < depth_map[x, y]:
                    depth_map[x, y] = point_dist
                    rgb_map[x, y] = point_rgb

    # Set empty pixels to default empty value
    for x in range(depth_map.shape[0]):
        for y in range(depth_map.shape[1]):
            if depth_map[x, y] > r_max:
                depth_map[x, y] = empty

    # Restore the cropped areas
    cropped_map_top = np.full((img_size[0], crop_top), empty, np.float32)
    cropped_map_bottom = np.full((img_size[0], crop_bottom), empty, np.float32)
    depth_map = np.concatenate(
        (cropped_map_top, depth_map, cropped_map_bottom), axis=1)

    cropped_map_top = np.zeros((img_size[0], crop_top, 3), dtype=np.uint8)
    cropped_map_bottom = np.zeros((img_size[0], crop_bottom, 3), dtype=np.uint8)
    rgb_map = np.concatenate(
        (cropped_map_top, rgb_map, cropped_map_bottom), axis=1)

    return rgb_map, depth_map


# ----------------------------------------------------------------------

@njit(cache=True, nogil=True)
def splatting_index(
        xyz_to_img, indices, img_opk, img_mask=None, img_size=(1024, 512),
        crop_top=0, crop_bottom=0, voxel=0.1, r_max=30, r_min=0.5, growth_k=0.2,
        growth_r=10, empty=0, no_id=-1):
    # We store indices in int64 format so we only accept indices up to
    # np.iinfo(np.int64).max
    num_points = xyz_to_img.shape[0]
    if num_points >= 9223372036854775807:
        raise OverflowError

    # Rotation matrix from image Euler angle pose
    img_rotation = pose_to_rotation_matrix_cpu(img_opk)

    # Remove points outside of image range
    dist = norm_cpu(xyz_to_img)
    in_range = np.where(np.logical_and(r_min < dist, dist < r_max))[0]

    # Project points to float pixel coordinates
    x_proj, y_proj = equirectangular_projection_cpu(
        xyz_to_img[in_range], dist[in_range], img_rotation, img_size)

    # Remove points outside of camera field of view
    in_fov = field_of_view_cpu(
        x_proj, y_proj, crop_top, img_size[1] - crop_bottom, mask=img_mask)

    # Compute projection pixel patches sizes
    width_pix = array_pixel_width_cpu(
        y_proj[in_fov], dist[in_range][in_fov], img_size=img_size,
        voxel=voxel, k=growth_k, d=growth_r)
    pix_masks = pixel_masks_cpu(x_proj[in_fov], y_proj[in_fov], width_pix)
    pix_masks = border_pixel_masks_cpu(
        pix_masks, 0, img_size[0], crop_top, img_size[1] - crop_bottom)
    pix_masks[:, 2:] -= crop_top  # Remove y-crop offset

    # Cropped depth map initialization
    cropped_img_size = (img_size[0], img_size[1] - crop_bottom - crop_top)
    depth_map = np.full(cropped_img_size, r_max + 1, np.float32)

    # Indices map intitialization
    # We store indices in int64 so we assumes point indices are lower
    # than max int64 ~ 2.14 x 10^9.
    # We need the negative for empty pixels
    idx_map = np.full(cropped_img_size, no_id, dtype=np.int64)

    # Loop through indices for points in range and in FOV
    dist = dist[in_range][in_fov]
    indices = indices[in_range][in_fov]
    for i_point in range(dist.shape[0]):

        point_dist = dist[i_point]
        point_idx = indices[i_point]
        point_pix_mask = pix_masks[i_point]

        # Update maps where point is closest recorded
        x_a, x_b, y_a, y_b = point_pix_mask
        for x in range(x_a, x_b):
            for y in range(y_a, y_b):
                if point_dist < depth_map[x, y]:
                    depth_map[x, y] = point_dist
                    idx_map[x, y] = point_idx

    # Set empty pixels to default empty value
    for x in range(depth_map.shape[0]):
        for y in range(depth_map.shape[1]):
            if depth_map[x, y] > r_max:
                depth_map[x, y] = empty

    # Restore the cropped areas
    cropped_map_top = np.full((img_size[0], crop_top), empty, np.float32)
    cropped_map_bottom = np.full((img_size[0], crop_bottom), empty, np.float32)
    depth_map = np.concatenate(
        (cropped_map_top, depth_map, cropped_map_bottom), axis=1)

    cropped_map_top = np.full((img_size[0], crop_top), no_id, np.int64)
    cropped_map_bottom = np.full((img_size[0], crop_bottom), no_id, np.int64)
    idx_map = np.concatenate(
        (cropped_map_top, idx_map, cropped_map_bottom), axis=1)

    return idx_map, depth_map


# ----------------------------------------------------------------------

@njit(cache=True, nogil=True)
def splatting(
        xyz_to_img, indices, img_opk, linearity=None, planarity=None,
        scattering=None, normals=None, img_mask=None, img_size=(1024, 512),
        crop_top=0, crop_bottom=0, voxel=0.1, r_max=30, r_min=0.5, growth_k=0.2,
        growth_r=10, empty=0, no_id=-1, exact=False):
    # We store indices in int64 format so we only accept indices up to
    # np.iinfo(np.int64).max
    num_points = xyz_to_img.shape[0]
    if num_points >= 9223372036854775807:
        raise OverflowError

    # Initialize pointwise geometric features to zero if not provided
    if linearity is None:
        linearity = np.zeros(xyz_to_img.shape[0], dtype=np.float32)
    if planarity is None:
        planarity = np.zeros(xyz_to_img.shape[0], dtype=np.float32)
    if scattering is None:
        scattering = np.zeros(xyz_to_img.shape[0], dtype=np.float32)
    if normals is None:
        normals = np.zeros((xyz_to_img.shape[0], 3), dtype=np.float32)
    linearity = linearity.astype(np.float32)
    planarity = planarity.astype(np.float32)
    scattering = scattering.astype(np.float32)
    normals = normals.astype(np.float32)

    # Rotation matrix from image Euler angle pose
    img_rotation = pose_to_rotation_matrix_cpu(img_opk)

    # Remove points outside of image range
    dist = norm_cpu(xyz_to_img)
    in_range = np.where(np.logical_and(r_min < dist, dist < r_max))[0]
    xyz_to_img = xyz_to_img[in_range]
    dist = dist[in_range]
    indices = indices[in_range]
    linearity = linearity[in_range]
    planarity = planarity[in_range]
    scattering = scattering[in_range]
    normals = normals[in_range]

    # Project points to float pixel coordinates
    x_proj, y_proj = equirectangular_projection_cpu(
        xyz_to_img, dist, img_rotation, img_size)

    # Remove points outside of camera field of view
    in_fov = field_of_view_cpu(
        x_proj, y_proj, crop_top, img_size[1] - crop_bottom, mask=img_mask)
    xyz_to_img = xyz_to_img[in_fov]
    dist = dist[in_fov]
    indices = indices[in_fov]
    linearity = linearity[in_fov]
    planarity = planarity[in_fov]
    scattering = scattering[in_fov]
    normals = normals[in_fov]
    x_proj = x_proj[in_fov]
    y_proj = y_proj[in_fov]

    # Compute projection pixel patches sizes
    width_pix = array_pixel_width_cpu(
        y_proj, dist, img_size=img_size, voxel=voxel, k=growth_k,
        d=growth_r)
    pix_masks = pixel_masks_cpu(x_proj, y_proj, width_pix)
    pix_masks = border_pixel_masks_cpu(
        pix_masks, 0, img_size[0], crop_top, img_size[1] - crop_bottom)
    pix_masks[:, 2:] -= crop_top  # Remove y-crop offset

    # Compute the N x F array of pointwise projection features carrying:
    #     - normalized depth
    #     - linearity
    #     - planarity
    #     - scattering
    #     - orientation to the surface
    #     - normalized pixel height
    depth = normalize_dist_cpu(dist, low=r_min, high=r_max)
    orientation = orientation_cpu(
        xyz_to_img / (dist + 1e-4).reshape((-1, 1)), normals)
    height = normalize_height_cpu(y_proj, img_size[1])
    features = np.column_stack(
        (depth, linearity, planarity, scattering, orientation, height))
    n_feat = features.shape[1]

    # Cropped depth map initialization
    cropped_img_size = (img_size[0], img_size[1] - crop_bottom - crop_top)
    depth_map = np.full(cropped_img_size, r_max + 1, dtype=np.float32)

    # Cropped indices map initialization
    # We store indices in int64 so we assumes point indices are lower
    # than max int64 ~ 2.14 x 10^9.
    # We need the negative for empty pixels
    idx_map = np.full(cropped_img_size, no_id, dtype=np.int64)

    # Cropped feature map initialization
    feat_map = np.zeros((*cropped_img_size, n_feat), dtype=np.float32)

    # Loop through indices for points in range and in FOV
    for i_point in range(dist.shape[0]):

        point_dist = dist[i_point]
        point_idx = indices[i_point]
        point_pix_mask = pix_masks[i_point]
        point_feat = features[i_point]

        # Update maps where point is closest recorded
        x_a, x_b, y_a, y_b = point_pix_mask
        for x in range(x_a, x_b):
            for y in range(y_a, y_b):
                if point_dist < depth_map[x, y]:
                    depth_map[x, y] = point_dist
                    if exact:
                        # Store the local indices if 'exact=True' mode.
                        # These indices can then be used to efficiently
                        # build the 'exact' maps without the need for
                        # 'np.isin', which is not supported un numba.
                        idx_map[x, y] = i_point
                    else:
                        # Store the real point indices otherwise
                        idx_map[x, y] = point_idx
                        feat_map[x, y] = point_feat

    # When 'exact=True', we use the results from the previous projection
    # to extract the seen points. The output maps are sparse, as seen
    # points are only mapped to the center of their splats, without
    # artificially-built splatting masks.
    if exact:
        # Recover the local indices of seen points
        idx_seen = np.unique(idx_map)
        idx_seen = idx_seen[idx_seen != no_id]

        # Reinitialize the output maps
        depth_map = np.full(cropped_img_size, r_max + 1, dtype=np.float32)
        idx_map = np.full(cropped_img_size, no_id, dtype=np.int64)

        # Convert the pixel projection coordinates to int
        x_proj = x_proj.astype(np.int32)
        y_proj = y_proj.astype(np.int32)

        # Loop through the seen points and populate only the center of
        # splats in the maps. We can update maps without worrying about
        # occlusions here.
        if idx_seen.shape[0] > 0:
            for i_point in idx_seen:
                point_dist = dist[i_point]
                point_idx = indices[i_point]
                point_feat = features[i_point]
                x = x_proj[i_point]
                y = y_proj[i_point]

                # Update maps without worrying about occlusions here
                depth_map[x, y] = point_dist
                idx_map[x, y] = point_idx
                feat_map[x, y] = point_feat

    # Set empty pixels to default empty value
    for x in range(depth_map.shape[0]):
        for y in range(depth_map.shape[1]):
            if depth_map[x, y] > r_max:
                depth_map[x, y] = empty

    # Restore the cropped areas
    cropped_map_top = np.full((img_size[0], crop_top), empty, np.float32)
    cropped_map_bottom = np.full((img_size[0], crop_bottom), empty, np.float32)
    depth_map = np.concatenate(
        (cropped_map_top, depth_map, cropped_map_bottom), axis=1)

    cropped_map_top = np.full((img_size[0], crop_top), no_id, np.int64)
    cropped_map_bottom = np.full((img_size[0], crop_bottom), no_id, np.int64)
    idx_map = np.concatenate(
        (cropped_map_top, idx_map, cropped_map_bottom), axis=1)

    cropped_map_top = np.zeros((img_size[0], crop_top, n_feat), np.float32)
    cropped_map_bottom = np.zeros((img_size[0], crop_bottom, n_feat), np.float32)
    feat_map = np.concatenate(
        (cropped_map_top, feat_map, cropped_map_bottom), axis=1)

    return idx_map, depth_map, feat_map


# TODO : all-torch GPU-parallelized projection ? Rather than iteratively
#  populating the depth map, create a set of target pixel coordinates and
#  associated meta-data (dist, point ID, normal orientation, ...). Then
#  use torch-GPU operations to extract the meta-data with the smallest
#  dist for each pixel coordinate ? This seems possible with torch.scatter_.
#  To understand how, see:
#    - torch_points3d.utils.multimodal.lexargunique
#    - https://medium.com/@yang6367/understand-torch-scatter-b0fd6275331c


# ----------------------------------------------------------------------

@torch_to_numba
@njit(cache=True, nogil=True)
def project_cpu(
        xyz_to_img, img_opk, img_mask=None, img_size=(1024, 512),
        crop_top=0, crop_bottom=0, r_max=30, r_min=0.5):
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
    in_range = torch.where(
        torch.logical_and(r_min < dist, dist < r_max))[0]
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
    """
    # Compute splatting masks for equirectangular images
    splat = equirectangular_splat_cpu(
        x_proj, y_proj, dist, img_size=img_size, crop_top=crop_top,
        crop_bottom=crop_bottom, voxel=voxel, k=growth_k, d=growth_r)

    # Cropped depth map initialization
    d_max = dist.max() + 1
    cropped_img_size = (img_size[0], img_size[1] - crop_bottom - crop_top)
    depth_map = np.full(cropped_img_size, d_max + 1, dtype=np.float32)

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
                y = y_proj[i_point]
                idx_map[x, y] = i_point

    # Recover final point indices and corresponding pixel coordinates
    x_pix, y_pix = np.where(idx_map != no_id)
    indices = np.zeros_like(x_pix)
    for i, (x, y) in enumerate(zip(x_pix, y_pix)):
        indices[i] = idx_map[x, y]

    return indices, x_pix, y_pix


def splatting_cuda(
        x_proj, y_proj, dist, img_size=(1024, 512), crop_top=0,
        crop_bottom=0, voxel=0.1, growth_k=0.2, growth_r=10, exact=False):
    """Compute visibility model with splatting on the GPU with torch and
    cuda.
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
        x_pix = torch.round(x_proj[indices]).long()
        y_pix = torch.round(y_proj[indices]).long()

    return indices, x_pix, y_pix


# **********************************************************************

from abc import ABC


class VisibilityModel(ABC):

    def __init__(
            self, xyz_to_img, indices, img_opk, linearity=None, planarity=None,
            scattering=None, normals=None, img_mask=None, img_depth=None,
            img_size=(1024, 512), crop_top=0, crop_bottom=0, voxel=0.1,
            r_max=30, r_min=0.5, growth_k=0.2, growth_r=10, empty=0, no_id=-1,
            exact=False, use_cuda=False):
        self.xyz_to_img = xyz_to_img
        self.indices = indices
        self.img_opk = img_opk
        self.linearity = linearity
        self.planarity = planarity
        self.scattering = scattering
        self.img_mask = img_mask
        self.img_depth = img_depth
        self.img_size = img_size
        self.crop_bottom = crop_bottom
        self.r_max = r_max
        self.r_min = r_min
        self.growth_k = growth_k
        self.growth_r = growth_r
        self.no_id = no_id
        self.exact = exact
        self.use_cuda = use_cuda and torch.cuda.is_available()

    def preprocess(self):
        if self.use_cuda:
            idx_proj, dist, x_proj, y_proj = project_cuda(
                self.xyz_to_img, self.img_opk, img_mask=self.img_mask,
                img_size=self.img_size, crop_top=self.crop_top,
                crop_bottom=self.crop_bottom, r_max=self.r_max,
                r_min=self.r_min)
        else:
            idx_proj, dist, x_proj, y_proj = project_cpu(
                self.xyz_to_img, self.img_opk, img_mask=self.img_mask,
                img_size=self.img_size, crop_top=self.crop_top,
                crop_bottom=self.crop_bottom, r_max=self.r_max,
                r_min=self.r_min)
        return
