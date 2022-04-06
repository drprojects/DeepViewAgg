import numpy as np
from numba import njit
import torch
import torch_scatter
from PIL import Image
from pykeops.torch import LazyTensor
from abc import ABC, abstractmethod


# -------------------------------------------------------------------- #
#                            Numba defaults                            #
# -------------------------------------------------------------------- #
OPK_DEFAULT=np.zeros(3, dtype=np.float32)
PINHOLE_DEFAULT=np.eye(4, dtype=np.float32)
FISHEYE_DEFAULT=np.ones(7, dtype=np.float32)
EXTRINSIC_DEFAULT=np.eye(4, dtype=np.float32)

# -------------------------------------------------------------------- #
#                             Numba wrapper                            #
# -------------------------------------------------------------------- #

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
        xyz_to_img, radius, img_opk, img_size):
    """Compute the projection of 3D points into the image pixel
    coordinate system of an equirectangular camera on the CPU with
    numpy and numba.

    :param xyz_to_img:
    :param radius:
    :param img_opk:
    :param img_size:
    :return:
    """
    # Rotation matrix from image Euler angle pose
    rotation = pose_to_rotation_matrix_cpu(img_opk)

    # Convert point to camera coordinate system
    v = xyz_to_img.dot(rotation.transpose())

    # Equirectangular projection
    t = np.arctan2(v[:, 1], v[:, 0])
    p = np.arccos(v[:, 2] / radius)

    # Angles to pixel position
    width, height = img_size
    w_pix = ((width - 1) * (1 - t / np.pi) / 2) % width
    h_pix = ((height - 1) * p / np.pi) % height

    # Nan values may appear in extreme cases, set them to zero
    w_pix[np.where(np.isnan(w_pix))] = 0
    h_pix[np.where(np.isnan(h_pix))] = 0

    return w_pix, h_pix


def equirectangular_projection_cuda(
        xyz_to_img, radius, img_opk, img_size):
    """Compute the projection of 3D points into the image pixel
    coordinate system of an equirectangular camera on the GPU with
    torch and cuda.

    :param xyz_to_img:
    :param radius:
    :param img_opk:
    :param img_size:
    :return:
    """
    # Rotation matrix from image Euler angle pose
    rotation = pose_to_rotation_matrix_cuda(img_opk)

    # Convert point to camera coordinate system
    v = xyz_to_img.mm(rotation.t())

    # Equirectangular projection
    t = torch.atan2(v[:, 1], v[:, 0])
    p = torch.acos(v[:, 2] / radius)

    # Angles to pixel position
    width, height = img_size
    w_pix = ((width - 1) * (1 - t / np.pi) / 2) % width
    h_pix = ((height - 1) * p / np.pi) % height

    # Nan values may appear in extreme cases, set them to zero
    w_pix[torch.where(w_pix.isnan())] = 0
    h_pix[torch.where(h_pix.isnan())] = 0

    return w_pix, h_pix


@njit(cache=True, nogil=True)
def pinhole_projection_cpu(xyz, img_extrinsic, img_intrinsic_pinhole, camera='scannet'):
    """Compute the projection of 3D points into the image pixel
    coordinate system of a pinhole camera described by a 4x4 intrinsic
    and a 4x4 extrinsic parameters tensors. Computations are executed
    on CPU with numpy and numba.

    :param xyz:
    :param img_extrinsic:
    :param img_intrinsic_pinhole:
    :return:
    """
    # Recover the 4x4 camera-to-world matrix
    if camera == 'scannet':
        camera_to_world = np.linalg.inv(np.ascontiguousarray(img_extrinsic))
        T = camera_to_world[:3, 3].copy().reshape((3, 1))
        R = camera_to_world[:3, :3].copy()
        p = R @ xyz.T + T

    elif camera == 'kitti360_perspective':
        camera_to_world = img_extrinsic
        T = camera_to_world[:3, 3].copy().reshape((1, 3))
        R = camera_to_world[:3, :3].copy()
        p = R.T @ (xyz - T).T
    else:
        raise ValueError

    x = p[0] * img_intrinsic_pinhole[0][0] / p[2] + img_intrinsic_pinhole[0][2]
    y = p[1] * img_intrinsic_pinhole[1][1] / p[2] + img_intrinsic_pinhole[1][2]
    z = p[2]
    
    # Make sure you return floa64 like other projection_cpu functions 
    # for Numba to happily compile
    return x.astype(np.float64), y.astype(np.float64), z.astype(np.float64)


def pinhole_projection_cuda(xyz, img_extrinsic, img_intrinsic_pinhole, camera='scannet'):
    """Compute the projection of 3D points into the image pixel
    coordinate system of a pinhole camera described by a 4x4 intrinsic
    and a 4x4 extrinsic parameters tensors. Computations are executed
    on GPU with torch and cuda.

    :param xyz:
    :param img_extrinsic:
    :param img_intrinsic_pinhole:
    :return:
    """
    if camera == 'scannet':
        camera_to_world = torch.inverse(img_extrinsic)
        T = camera_to_world[:3, 3].view(3, 1)
        R = camera_to_world[:3, :3]
        p = R @ xyz.T + T

    elif camera == 'kitti360_perspective':
        camera_to_world = img_extrinsic
        T = camera_to_world[:3, 3].view(1, 3)
        R = camera_to_world[:3, :3]
        p = R.T @ (xyz - T).T

    else:
        raise ValueError

    x = p[0] * img_intrinsic_pinhole[0][0] / p[2] + img_intrinsic_pinhole[0][2]
    y = p[1] * img_intrinsic_pinhole[1][1] / p[2] + img_intrinsic_pinhole[1][2]
    z = p[2]

    return x, y, z


@njit(cache=True, nogil=True)
def fisheye_projection_cpu(
        xyz, img_extrinsic, img_intrinsic_fisheye, camera='kitti360_fisheye'):
    """Compute the projection of 3D points into the image pixel
    coordinate system of a fisheye camera described by 6 intrinsic
    and a 4x4 extrinsic parameters tensors. Computations are executed
    on CPU with numpy and numba.

    Credit: https://github.com/autonomousvision/kitti360Scripts

    :param xyz:
    :param img_extrinsic:
    :param img_intrinsic_fisheye:
    :param camera:
    :return:
    """
    if camera == 'kitti360_fisheye':
        camera_to_world = img_extrinsic
        T = camera_to_world[:3, 3].copy().reshape((1, 3))
        R = camera_to_world[:3, :3].copy()
        p = R.T @ (xyz - T).T
    else:
        raise ValueError

    # Recover fisheye camera intrinsic parameters
    xi = img_intrinsic_fisheye[0]
    k1 = img_intrinsic_fisheye[1]
    k2 = img_intrinsic_fisheye[2]
    gamma1 = img_intrinsic_fisheye[3]
    gamma2 = img_intrinsic_fisheye[4]
    u0 = img_intrinsic_fisheye[5]
    v0 = img_intrinsic_fisheye[6]

    # Compute float pixel coordinates
    p = p.T
    norm = norm_cpu(p)
    
    x = p[:, 0] / (norm + 1e-4)
    y = p[:, 1] / (norm + 1e-4)
    z = p[:, 2] / (norm + 1e-4)

    x /= z + xi
    y /= z + xi

    r2 = x ** 2 + y ** 2
    r4 = r2 ** 2

    x = gamma1 * (1 + k1 * r2 + k2 * r4) * x + u0
    y = gamma2 * (1 + k1 * r2 + k2 * r4) * y + v0
    z = norm * p[:, 2] / np.abs(p[:, 2] + 1e-4)

    return x.astype(np.float64), y.astype(np.float64), z.astype(np.float64)


def fisheye_projection_cuda(
        xyz, img_extrinsic, img_intrinsic_fisheye, camera='kitti360_fisheye'):
    """Compute the projection of 3D points into the image pixel
    coordinate system of a fisheye camera described by 6 intrinsic
    and a 4x4 extrinsic parameters tensors. Computations are executed
    on GPU with torch and cuda.

    Credit: https://github.com/autonomousvision/kitti360Scripts

    :param xyz:
    :param img_extrinsic:
    :param img_intrinsic_fisheye:
    :param camera:
    :return:
    """
    if camera == 'kitti360_fisheye':
        camera_to_world = img_extrinsic
        T = camera_to_world[:3, 3].view(1, 3)
        R = camera_to_world[:3, :3]
        p = R.T @ (xyz - T).T

    else:
        raise ValueError

    # Recover fisheye camera intrinsic parameters
    xi = img_intrinsic_fisheye[0]
    k1 = img_intrinsic_fisheye[0]
    k2 = img_intrinsic_fisheye[0]
    gamma1 = img_intrinsic_fisheye[0]
    gamma2 = img_intrinsic_fisheye[0]
    u0 = img_intrinsic_fisheye[0]
    v0 = img_intrinsic_fisheye[0]

    # Compute float pixel coordinates
    p = p.T
    norm = torch.linalg.norm(p, dim=1)

    x = p[:, 0] / norm
    y = p[:, 1] / norm
    z = p[:, 2] / norm

    x /= z + xi
    y /= z + xi

    r2 = x**2 + y**2
    r4 = r2**2

    x = gamma1 * (1 + k1 * r2 + k2 * r4) * x + u0
    y = gamma2 * (1 + k1 * r2 + k2 * r4) * y + v0

    return x, y, norm * p[:, 2] / (p[:, 2].abs() + 1e-4)


@njit(cache=True, nogil=True, fastmath=True)
def field_of_view_cpu(
        x_pix, y_pix, x_min=None, x_max=None, y_min=None, y_max=None, z=None,
        img_mask=None):
    """

    :param x_pix:
    :param y_pix:
    :param x_min:
    :param x_max:
    :param y_min:
    :param y_max:
    :param z:
    :param img_mask:
    :return:
    """
    in_fov = np.ones_like(x_pix, dtype=np.bool_)

    if x_min is not None:
        in_fov *= (x_min <= x_pix)

    if y_min is not None:
        in_fov *= (y_min <= y_pix)

    if x_max is not None:
        in_fov *= (x_pix < x_max)

    if y_max is not None:
        in_fov *= (y_pix < y_max)

    if z is not None:
        in_fov *= (0 < z)

    if not img_mask is None:
        n_points = x_pix.shape[0]
        x_int = np.floor(x_pix).astype(np.uint32)
        y_int = np.floor(y_pix).astype(np.uint32)
        for i in range(n_points):
            if in_fov[i] and not img_mask[x_int[i], y_int[i]]:
                in_fov[i] = False
    return np.where(in_fov)[0]


def field_of_view_cuda(
        x_pix, y_pix, x_min=None, x_max=None, y_min=None, y_max=None, z=None,
        img_mask=None):
    """

    :param x_pix:
    :param y_pix:
    :param x_min:
    :param x_max:
    :param y_min:
    :param y_max:
    :param z_min:
    :param z_max:
    :param img_mask:
    :return:
    """
    in_fov = torch.ones_like(x_pix).bool()

    if x_min is not None:
        in_fov *= (x_min <= x_pix)

    if y_min is not None:
        in_fov *= (y_min <= y_pix)

    if x_max is not None:
        in_fov *= (x_pix < x_max)

    if y_max is not None:
        in_fov *= (y_pix < y_max)

    if z is not None:
        in_fov *= (0 < z)

    if not img_mask is None:
        x_int = torch.floor(x_pix).long()
        y_int = torch.floor(y_pix).long()
        in_fov = torch.logical_and(in_fov, img_mask[x_int, y_int])
    return torch.where(in_fov)[0]


@torch_to_numba
@njit(cache=True, nogil=True)
def camera_projection_cpu(
        xyz, img_xyz, img_opk=None, img_intrinsic_pinhole=None,
        img_intrinsic_fisheye=None, img_extrinsic=None, img_mask=None, 
        img_size=(1024, 512), crop_top=0, crop_bottom=0, r_max=30, r_min=0.5,
        camera='s3dis_equirectangular'):
    assert img_mask is None or img_mask.shape == img_size
    
    # Need to set defaults inside the function rather than in the kwargs
    # because those will be overwritten by the parent CPU-GPU dispatcher
    # function
    if img_opk is None:
        img_opk = OPK_DEFAULT
    if img_intrinsic_pinhole is None:
        img_intrinsic_pinhole = PINHOLE_DEFAULT
    if img_intrinsic_fisheye is None:
        img_intrinsic_fisheye = FISHEYE_DEFAULT
    if img_extrinsic is None:
        img_extrinsic = EXTRINSIC_DEFAULT
    
    # We store indices in int64 format so we only accept indices up to
    # np.iinfo(np.int64).max
    num_points = xyz.shape[0]
    if num_points >= 9223372036854775807:
        raise OverflowError
    
    # Initialize the indices to keep track of selected points
    indices = np.arange(num_points)

    # Remove points outside of image range
    dist = norm_cpu(xyz - img_xyz)
    in_range = np.where(np.logical_and(r_min < dist, dist < r_max))[0]
    xyz = xyz[in_range]
    dist = dist[in_range]
    indices = indices[in_range]
    
    # Project points to float pixel coordinates
    if camera in ['kitti360_perspective', 'scannet']:
        x_proj, y_proj, z_proj = pinhole_projection_cpu(
            xyz, img_extrinsic, img_intrinsic_pinhole, camera=camera)
    elif camera == 'kitti360_fisheye':
        x_proj, y_proj, z_proj = fisheye_projection_cpu(
            xyz, img_extrinsic, img_intrinsic_fisheye, camera=camera)
    elif camera == 's3dis_equirectangular' and img_opk is not None:
        x_proj, y_proj = equirectangular_projection_cpu(
            xyz - img_xyz, dist, img_opk, img_size)
        z_proj = np.ones_like(x_proj)
    else:
        raise ValueError

    # Remove points outside of camera field of view
    in_fov = field_of_view_cpu(
        x_proj, y_proj, x_min=0, x_max=img_size[0], y_min=crop_top,
        y_max=img_size[1] - crop_bottom, z=z_proj, img_mask=img_mask)
    dist = dist[in_fov]
    indices = indices[in_fov]
    x_proj = x_proj[in_fov]
    y_proj = y_proj[in_fov]

    return indices, dist, x_proj, y_proj


def camera_projection_cuda(
        xyz, img_xyz, img_opk=None, img_intrinsic_pinhole=None,
        img_intrinsic_fisheye=None, img_extrinsic=None,
        img_mask=None, img_size=(1024, 512), crop_top=0, crop_bottom=0,
        r_max=30, r_min=0.5, camera='s3dis_equirectangular'):
    assert img_mask is None or img_mask.shape == img_size, \
        f'Expected img_mask to be a torch.BoolTensor of shape ' \
        f'img_size={img_size} but got size={img_mask.shape}.'

    # We store indices in int64 format so we only accept indices up to
    # torch.iinfo(torch.long).max
    num_points = xyz.shape[0]
    if num_points >= 9223372036854775807:
        raise OverflowError

    # Initialize the indices to keep track of selected points
    indices = torch.arange(num_points, device=xyz.device)

    # Remove points outside of image range
    dist = norm_cuda(xyz - img_xyz)
    in_range = torch.where(torch.logical_and(r_min < dist, dist < r_max))[0]
    xyz = xyz[in_range]
    dist = dist[in_range]
    indices = indices[in_range]

    # Project points to float pixel coordinates
    if camera in ['kitti360_perspective', 'scannet']:
        x_proj, y_proj, z_proj = pinhole_projection_cuda(
            xyz, img_extrinsic, img_intrinsic_pinhole, camera=camera)
    elif camera == 'kitti360_fisheye':
        x_proj, y_proj, z_proj = fisheye_projection_cuda(
            xyz, img_extrinsic, img_intrinsic_fisheye, camera=camera)
    elif camera == 's3dis_equirectangular' and img_opk is not None:
        x_proj, y_proj = equirectangular_projection_cuda(
            xyz - img_xyz, dist, img_opk, img_size)
        z_proj = None
    else:
        raise ValueError

    # Remove points outside of camera field of view
    in_fov = field_of_view_cuda(
        x_proj, y_proj, x_min=0, x_max=img_size[0], y_min=crop_top,
        y_max=img_size[1] - crop_bottom, z=z_proj, img_mask=img_mask)
    dist = dist[in_fov]
    indices = indices[in_fov]
    x_proj = x_proj[in_fov]
    y_proj = y_proj[in_fov]

    return indices, dist, x_proj, y_proj


def camera_projection(
        xyz, img_xyz, img_opk=None, img_intrinsic_pinhole=None,
        img_intrinsic_fisheye=None, img_extrinsic=None, img_mask=None,
        img_size=(1024, 512), crop_top=0, crop_bottom=0, r_max=30, r_min=0.5,
        camera='s3dis_equirectangular', **kwargs):
    """

    :param xyz:
    :param img_xyz:
    :param img_opk:
    :param img_intrinsic_pinhole:
    :param img_extrinsic:
    :param img_mask:
    :param img_size:
    :param crop_top:
    :param crop_bottom:
    :param r_max:
    :param r_min:
    :param kwargs:
    :return:
    """
    assert img_mask is None or img_mask.shape == img_size, \
        f'Expected img_mask to be a torch.BoolTensor of shape ' \
        f'img_size={img_size} but got size={img_mask.shape}.'
    
    f = camera_projection_cuda if xyz.is_cuda else camera_projection_cpu
    return f(
        xyz, img_xyz, img_opk=img_opk, img_intrinsic_pinhole=img_intrinsic_pinhole,
        img_intrinsic_fisheye=img_intrinsic_fisheye,
        img_extrinsic=img_extrinsic, img_mask=img_mask, img_size=img_size,
        crop_top=crop_top, crop_bottom=crop_bottom, r_max=r_max, r_min=r_min,
        camera=camera)


# -------------------------------------------------------------------- #
#                     Visibility Method - Splatting                    #
# -------------------------------------------------------------------- #

@njit(cache=True, nogil=True)
def equirectangular_splat_cpu(
        x_proj, y_proj, dist, img_size=(1024, 512), crop_top=0, crop_bottom=0,
        voxel=0.02, k_swell=1.0, d_swell=1000):
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
        voxel=0.02, k_swell=1.0, d_swell=1000):
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


@njit(cache=True, nogil=True)
def pinhole_splat_cpu(
        x_proj, y_proj, dist, img_intrinsic_pinhole, img_size=(1024, 512), crop_top=0,
        crop_bottom=0, voxel=0.02, k_swell=1.0, d_swell=1000):
    """

    :param x_proj:
    :param y_proj:
    :param dist:
    :param img_intrinsic_pinhole:
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
    swell = (1 + k_swell * np.exp(-dist / np.log(d_swell))) * voxel / dist
    width_x = swell * img_intrinsic_pinhole[0, 0]
    width_y = swell * img_intrinsic_pinhole[1, 1]

    # NB: stack+transpose faster than column stack
    splat_xy_width = np.stack((width_x, width_y)).transpose()

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


def pinhole_splat_cuda(
        x_proj, y_proj, dist, img_intrinsic_pinhole, img_size=(1024, 512), crop_top=0,
        crop_bottom=0, voxel=0.02, k_swell=1.0, d_swell=1000):
    """

    :param x_proj:
    :param y_proj:
    :param dist:
    :param img_intrinsic_pinhole:
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
    swell = (1 + k_swell * torch.exp(-dist / np.log(d_swell))) * voxel / dist
    width_x = swell * img_intrinsic_pinhole[0, 0]
    width_y = swell * img_intrinsic_pinhole[1, 1]
    splat_xy_width = torch.stack((width_x, width_y)).t()

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


@njit(cache=True, nogil=True)
def fisheye_splat_cpu(
        x_proj, y_proj, xyz, img_extrinsic, img_intrinsic_fisheye,
        img_size=(1024, 512), crop_top=0, crop_bottom=0, voxel=0.02,
        k_swell=1.0, d_swell=1000, camera='kitti360_fisheye'):
    """

    :param x_proj:
    :param y_proj:
    :param xyz:
    :param img_extrinsic:
    :param img_intrinsic_fisheye:
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
    dist = norm_cpu(xyz)
    swell = (1 + k_swell * np.exp(-dist / np.log(d_swell)))

    # Compute the projection of the top of the voxel / cube. The
    # distance between this projection and the actual point's projection
    # will serve as a proxy to estimate the splat's size. This heuristic
    # does not hold if voxels are quite large and close to the camera,
    # this should not cause too much trouble for outdoor scenes but may
    # affect narrow indoor scenes with close-by objects such as walls
    # TODO: improve fisheye splat computation
    z_offset = np.zeros_like(xyz)
    z_offset[:, 2] = swell * voxel / 2
    x, y, _ = fisheye_projection_cpu(
        xyz + z_offset, img_extrinsic, img_intrinsic_fisheye, camera=camera)
    splat_xy_width = 2 * np.sqrt((x_proj - x)**2 + (y_proj - y)**2).repeat(2).reshape(-1, 2)

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


def fisheye_splat_cuda(
        x_proj, y_proj, xyz, img_extrinsic, img_intrinsic_fisheye, img_size=(1024, 512), crop_top=0,
        crop_bottom=0, voxel=0.02, k_swell=1.0, d_swell=1000, camera='kitti360_fisheye'):
    """

    :param x_proj:
    :param y_proj:
    :param xyz:
    :param img_extrinsic:
    :param img_intrinsic_fisheye:
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
    dist = norm_cuda(xyz)
    swell = (1 + k_swell * torch.exp(-dist / np.log(d_swell)))

    # Compute the projection of the top of the voxel / cube. The
    # distance between this projection and the actual point's projection
    # will serve as a proxy to estimate the splat's size. This heuristic
    # does not hold if voxels are quite large and close to the camera,
    # this should not cause too much trouble for outdoor scenes but may
    # affect narrow indoor scenes with close-by objects such as walls
    # TODO: improve fisheye splat computation
    z_offset = torch.zeros_like(xyz)
    z_offset[:, 2] = swell * voxel / 2
    x, y, _ = fisheye_projection_cuda(xyz + z_offset, img_extrinsic, img_intrinsic_fisheye, camera=camera)
    splat_xy_width = 2 * ((x_proj - x) ** 2 + (y_proj - y) ** 2).sqrt().repeat(2, 1).t()

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


@torch_to_numba
@njit(cache=True, nogil=True)
def visibility_from_splatting_cpu(
        x_proj, y_proj, dist, xyz, img_extrinsic=None, 
        img_intrinsic_pinhole=None, img_intrinsic_fisheye=None, 
        img_size=(1024, 512), crop_top=0, crop_bottom=0, voxel=0.1, k_swell=1.0,
        d_swell=1000, exact=False, camera='s3dis_equirectangular'):
    """Compute visibility model with splatting on the CPU with numpy and
    numba.

    Although top and bottom cropping can be specified, the returned
    coordinates are expressed in the non-cropped image pixel coordinate
    system.

    :param x_proj:
    :param y_proj:
    :param dist:
    :param img_extrinsic:
    :param img_intrinsic_pinhole:
    :param img_intrinsic_fisheye:
    :param img_size:
    :param crop_top:
    :param crop_bottom:
    :param voxel:
    :param k_swell:
    :param d_swell:
    :param exact:
    :param camera:
    :return:
    """
    assert x_proj.shape[0] == y_proj.shape[0] == dist.shape[0] > 0
    
    # Need to set defaults inside the function rather than in the kwargs
    # because those will be overwritten by the parent CPU-GPU dispatcher
    # function
    if img_intrinsic_pinhole is None:
        img_intrinsic_pinhole = PINHOLE_DEFAULT
    if img_intrinsic_fisheye is None:
        img_intrinsic_fisheye = FISHEYE_DEFAULT
    if img_extrinsic is None:
        img_extrinsic = EXTRINSIC_DEFAULT

    # Compute splatting masks
    if camera == 's3dis_equirectangular':
        splat = equirectangular_splat_cpu(
            x_proj, y_proj, dist, img_size=img_size, crop_top=crop_top,
            crop_bottom=crop_bottom, voxel=voxel, k_swell=k_swell,
            d_swell=d_swell)
    elif camera in ['kitti360_perspective', 'scannet']:
        splat = pinhole_splat_cpu(
            x_proj, y_proj, dist, img_intrinsic_pinhole, img_size=img_size,
            crop_top=crop_top, crop_bottom=crop_bottom, voxel=voxel,
            k_swell=k_swell, d_swell=d_swell)
    elif camera == 'kitti360_fisheye':
        splat = fisheye_splat_cpu(
            x_proj, y_proj, xyz, img_extrinsic, img_intrinsic_fisheye,
            img_size=img_size, crop_top=crop_top, crop_bottom=crop_bottom,
            voxel=voxel, k_swell=k_swell, d_swell=d_swell, camera=camera)
    else:
        raise ValueError

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
        x_proj, y_proj, dist, xyz, img_extrinsic=None, img_intrinsic_pinhole=None,
        img_intrinsic_fisheye=None, img_size=(1024, 512),
        crop_top=0, crop_bottom=0, voxel=0.1, k_swell=1.0, d_swell=1000,
        exact=False, camera='s3dis_equirectangular'):
    """Compute visibility model with splatting on the GPU with torch and
    cuda.

    Although top and bottom cropping can be specified, the returned
    coordinates are expressed in the non-cropped image pixel coordinate
    system.

    :param x_proj:
    :param y_proj:
    :param dist:
    :param img_extrinsic:
    :param img_intrinsic_pinhole:
    :param img_intrinsic_fisheye:
    :param img_size:
    :param crop_top:
    :param crop_bottom:
    :param voxel:
    :param k_swell:
    :param d_swell:
    :param exact:
    :param camera:
    :param kwargs:
    :return:
    """
    assert x_proj.shape[0] == y_proj.shape[0] == dist.shape[0] > 0

    # Initialization
    device = x_proj.device
    n_points = x_proj.shape[0]

    # Compute splatting masks
    if camera == 's3dis_equirectangular':
        splat = equirectangular_splat_cuda(
            x_proj, y_proj, dist, img_size=img_size, crop_top=crop_top,
            crop_bottom=crop_bottom, voxel=voxel, k_swell=k_swell,
            d_swell=d_swell)
    elif camera in ['kitti360_perspective', 'scannet']:
        splat = pinhole_splat_cuda(
            x_proj, y_proj, dist, img_intrinsic_pinhole, img_size=img_size,
            crop_top=crop_top, crop_bottom=crop_bottom, voxel=voxel,
            k_swell=k_swell, d_swell=d_swell)
    elif camera == 'kitti360_fisheye':
        splat = fisheye_splat_cuda(
            x_proj, y_proj, xyz, img_extrinsic, img_intrinsic_fisheye,
            img_size=img_size, crop_top=crop_top, crop_bottom=crop_bottom,
            voxel=voxel, k_swell=k_swell, d_swell=d_swell, camera=camera)
    else:
        raise ValueError(f"Unknown camera '{camera}'")

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


def visibility_from_splatting(
        x_proj, y_proj, dist, xyz, img_extrinsic=None, img_intrinsic_pinhole=None,
        img_intrinsic_fisheye=None, img_size=(1024, 512),
        crop_top=0, crop_bottom=0, voxel=0.1, k_swell=1.0, d_swell=1000,
        exact=False, camera='s3dis_equirectangular', **kwargs):
    """

    :param x_proj:
    :param y_proj:
    :param dist:
    :param img_extrinsic:
    :param img_intrinsic_pinhole:
    :param img_intrinsic_fisheye:
    :param img_size:
    :param crop_top:
    :param crop_bottom:
    :param voxel:
    :param k_swell:
    :param d_swell:
    :param exact:
    :param camera:
    :param kwargs:
    :return:
    """
    f = visibility_from_splatting_cuda if x_proj.is_cuda \
        else visibility_from_splatting_cpu

    return f(
        x_proj, y_proj, dist, xyz, img_extrinsic=img_extrinsic,
        img_intrinsic_pinhole=img_intrinsic_pinhole,
        img_intrinsic_fisheye=img_intrinsic_fisheye, img_size=img_size,
        crop_top=crop_top, crop_bottom=crop_bottom, voxel=voxel,
        k_swell=k_swell, d_swell=d_swell, exact=exact,
        camera=camera)


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
    assert x_proj.shape[0] == y_proj.shape[0] == dist.shape[0] > 0

    # Read the depth map
    # TODO: only supports S3DIS-type depth map. Extend to other formats
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


def visibility_biasutti(
        x_proj, y_proj, dist, img_size=None, k=75,
        margin=None, threshold=None, **kwargs):
    """Compute visibility model based Biasutti et al. method as
    described in:

    "Visibility estimation in point clouds with variable density"
    Source: https://hal.archives-ouvertes.fr/hal-01812061/document

    :param x_proj:
    :param y_proj:
    :param dist:
    :param img_size:
    :param k:
    :param margin:
    :param threshold:
    :return:
    """
    assert x_proj.shape[0] == y_proj.shape[0] == dist.shape[0] > 0

    # Search k-nearest neighbors in the image pixel coordinate system
    neighbors = k_nn_image_system(
        x_proj, y_proj, k=k, x_margin=margin, x_width=img_size[0])

    # Compute the visibility and recover visible point indices
    dist_nn = dist[neighbors]
    dist_min = dist_nn.min(dim=1).values
    dist_max = dist_nn.max(dim=1).values
    alpha = torch.exp(-((dist - dist_min) / (dist_max - dist_min)) ** 2)
    if threshold is None:
        threshold = alpha.mean()
    indices = torch.where(alpha >= threshold)[0]

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

    orientation = (u * v).sum(dim=1).abs()
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
    features = []

    # Depth
    if dist is not None:
        features.append(normalize_dist_cuda(dist, low=r_min, high=r_max))
    # Linearity
    if linearity is not None:
        features.append(linearity)
    # Planarity
    if planarity is not None:
        features.append(planarity)
    # Scattering
    if scattering is not None:
        features.append(scattering)

    # Orientation to the normal
    if xyz_to_img is not None and dist is not None and normals is not None:
        features.append(orientation_cuda(
            xyz_to_img / (dist + 1e-4).reshape((-1, 1)), normals))

    # Pixel height
    if y_proj is not None:
        features.append((y_proj / img_size[1]).float())

    return torch.stack(features).t()


# -------------------------------------------------------------------- #
#                              Visibility                              #
# -------------------------------------------------------------------- #

def visibility(
        xyz, img_xyz, method='splatting', linearity=None, planarity=None,
        scattering=None, normals=None, use_cuda=True, **kwargs):
    """Compute the visibility of a point cloud with respect to a given
    camera pose.

    :param xyz:
    :param img_xyz:
    :param method:
    :param linearity:
    :param planarity:
    :param scattering:
    :param normals:
    :param use_cuda:
    :param kwargs:
    :return:
    """
    METHODS = ['splatting', 'depth_map', 'biasutti']
    assert method in METHODS, \
        f'Unknown method {method}, expected one of {METHODS}.'

    in_device = xyz.device
    if xyz.is_cuda:
        use_cuda = True
    elif not torch.cuda.is_available():
        use_cuda = False

    if use_cuda:
        xyz = xyz.cuda()
        img_xyz = img_xyz.cuda()
        linearity = linearity.cuda() if linearity is not None else None
        planarity = planarity.cuda() if planarity is not None else None
        scattering = scattering.cuda() if scattering is not None else None
        normals = normals.cuda() if normals is not None else None
        kwargs = {
            k: v.cuda() if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()}

    # Compute camera projection
    idx_1, dist, x_proj, y_proj = camera_projection(xyz, img_xyz, **kwargs)

    # Return if no projections are found
    if x_proj.shape[0] == 0:
        out = {}
        out['idx'] = torch.empty((0,), dtype=torch.long, device=in_device)
        out['x'] = torch.empty((0,), dtype=torch.long, device=in_device)
        out['y'] = torch.empty((0,), dtype=torch.long, device=in_device)
        out['depth'] = torch.empty((0,), dtype=torch.float, device=in_device)
        out['features'] = torch.empty((0,), dtype=torch.float, device=in_device)
        return out

    if method == 'splatting':
        idx_2, x_pix, y_pix = visibility_from_splatting(
            x_proj, y_proj, dist, xyz[idx_1], **kwargs)
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
    xyz = xyz[idx]
    dist = dist[idx_2]
    x_proj = x_proj[idx_2]
    y_proj = y_proj[idx_2]

    out = {}
    out['idx'] = idx.to(in_device)
    out['x'] = x_pix.to(in_device)
    out['y'] = y_pix.to(in_device)
    out['depth'] = dist.to(in_device)

    # Compute mapping features
    linearity = linearity[idx] if linearity is not None else None
    planarity = planarity[idx] if planarity is not None else None
    scattering = scattering[idx] if scattering is not None else None
    normals = normals[idx] if normals is not None else None
    out['features'] = postprocess_features(
        xyz - img_xyz, y_proj, dist, linearity, planarity, scattering,
        normals)

    return out


class VisibilityModel(ABC):

    def __init__(
            self, img_size=(1024, 512), crop_top=0, crop_bottom=0, r_max=30,
            r_min=0.5, camera='s3dis_equirectangular'):
        self.img_size = img_size
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
        self.r_max = r_max
        self.r_min = r_min
        self.camera = camera

    def _camera_projection(self, *args, **kwargs):
        return camera_projection(*args, **self.__dict__, **kwargs)

    @abstractmethod
    def _visibility(self):
        pass

    def _postprocess_features(self, *args):
        return postprocess_features(*args, **self.__dict__)

    def __call__(
            self, xyz, img_xyz, linearity=None, planarity=None, scattering=None,
            normals=None, **kwargs):
        """Compute the visibility of a point cloud with respect to a
        given camera pose.

        :param xyz:
        :param img_xyz:
        :param linearity:
        :param planarity:
        :param scattering:
        :param normals:
        :param kwargs:
        :return:
        """
        in_device = xyz.device

        # Compute camera projection
        idx_1, dist, x_proj, y_proj = self._camera_projection(
            xyz, img_xyz, **kwargs)

        # Return if no projections are found
        if x_proj.shape[0] == 0:
            out = {}
            out['idx'] = torch.empty((0,), dtype=torch.long, device=in_device)
            out['x'] = torch.empty((0,), dtype=torch.long, device=in_device)
            out['y'] = torch.empty((0,), dtype=torch.long, device=in_device)
            out['depth'] = torch.empty((0,), dtype=torch.float, device=in_device)
            out['features'] = torch.empty(
                (0,), dtype=torch.float, device=in_device)
            return out

        # Compute visibility of projected points
        idx_2, x_pix, y_pix = self._visibility(
            x_proj, y_proj, dist, xyz[idx_1], **kwargs)

        # Keep data only for mapped point
        idx = idx_1[idx_2]
        xyz = xyz[idx]
        dist = dist[idx_2]
        x_proj = x_proj[idx_2]
        y_proj = y_proj[idx_2]

        out = {}
        out['idx'] = idx
        out['x'] = x_pix
        out['y'] = y_pix
        out['depth'] = dist

        # Compute mapping features
        linearity = linearity[idx] if linearity is not None else None
        planarity = planarity[idx] if planarity is not None else None
        scattering = scattering[idx] if scattering is not None else None
        normals = normals[idx] if normals is not None else None
        out['features'] = self._postprocess_features(
            xyz - img_xyz, y_proj, dist, linearity, planarity, scattering,
            normals)

        return out

    def __repr__(self):
        attr_repr = ', '.join([f'{k}={v}' for k, v in self.__dict__.items()])
        return f'{self.__class__.__name__}({attr_repr})'


class SplattingVisibility(VisibilityModel, ABC):

    def __init__(
            self, voxel=0.1, k_swell=1.0, d_swell=1000, exact=False, **kwargs):
        super(SplattingVisibility, self).__init__(**kwargs)
        self.voxel = voxel
        self.k_swell = k_swell
        self.d_swell = d_swell
        self.exact = exact

    def _visibility(self, x_proj, y_proj, dist, xyz, **kwargs):
        return visibility_from_splatting(
            x_proj, y_proj, dist, xyz, **self.__dict__, **kwargs)


class DepthBasedVisibility(VisibilityModel, ABC):

    def __init__(self, depth_threshold=0.05, **kwargs):
        super(DepthBasedVisibility, self).__init__(**kwargs)
        self.depth_threshold = depth_threshold

    def _visibility(self, x_proj, y_proj, dist, xyz, **kwargs):
        return visibility_from_depth_map(
            x_proj, y_proj, dist, **self.__dict__, **kwargs)


class BiasuttiVisibility(VisibilityModel, ABC):

    def __init__(self, k=75, margin=None, threshold=None, **kwargs):
        super(BiasuttiVisibility, self).__init__(**kwargs)
        self.k = k
        self.margin = margin
        self.threshold = threshold

    def _visibility(self, x_proj, y_proj, dist, xyz, **kwargs):
        return visibility_biasutti(
            x_proj, y_proj, dist, **self.__dict__, **kwargs)

# TODO: support other depth map files formats. For now, only S3DIS format supported
