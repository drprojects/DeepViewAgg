import torch
import numpy as np
from numba import njit
from abc import ABC, abstractmethod


# -------------------------------------------------------------------- #
#                            Camera Helpers                            #
# -------------------------------------------------------------------- #

@njit(cache=True, nogil=True)
def opk_to_rotation_matrix_cpu(opk):
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


def opk_to_rotation_matrix_cuda(opk):
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


@njit(cache=True, nogil=True, fastmath=True)
def field_of_view_cpu(
        x_pix, y_pix, x_min=None, x_max=None, y_min=None, y_max=None, z=None,
        mask=None):
    """

    :param x_pix:
    :param y_pix:
    :param x_min:
    :param x_max:
    :param y_min:
    :param y_max:
    :param z:
    :param mask:
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

    if mask is not None:
        # If the mask comes from a CameraModel, may not be stored on the
        # same device as x_pix
        mask = mask.to(x_pix.device)
        n_points = x_pix.shape[0]
        x_int = np.floor(x_pix).astype(np.uint32)
        y_int = np.floor(y_pix).astype(np.uint32)
        for i in range(n_points):
            if in_fov[i] and not mask[x_int[i], y_int[i]]:
                in_fov[i] = False
    return np.where(in_fov)[0]


def field_of_view_cuda(
        x_pix, y_pix, x_min=None, x_max=None, y_min=None, y_max=None, z=None,
        mask=None):
    """

    :param x_pix:
    :param y_pix:
    :param x_min:
    :param x_max:
    :param y_min:
    :param y_max:
    :param z_min:
    :param z_max:
    :param mask:
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

    if mask is not None:
        x_int = torch.floor(x_pix).long()
        y_int = torch.floor(y_pix).long()
        in_fov = torch.logical_and(in_fov, mask[x_int, y_int])
    return torch.where(in_fov)[0]


# -------------------------------------------------------------------- #
#                                Camera                                #
# -------------------------------------------------------------------- #

class Camera(ABC):
    """Abstract class for camera models to inherit from. The camera
    model holds camera intrinsic parameters and projects 3D points at a
    given camera position using the `shoot` method. The camera may also
    produces voxel splats based on point projections through the `splat`
    method.

    To inherit from this class, camera models must implement
    the `project` and `splat` methods.

    Attributes
        size:tuple         size of the pixel grid
        mask:BoolTensor    boolean mask for point projection
    """

    def __init__(self, size=(1024, 512), mask=None):
        self.size = size
        self.mask = mask

    @property
    def mask(self):
        """Boolean mask used for 3D points projection in the images.

        If not None, must be a BoolTensor of size `self.size`.
        """
        return self._mask

    @mask.setter
    def mask(self, mask):
        if mask is None:
            self._mask = None
            return
        assert mask.dtype == torch.bool, \
            f"Expected a dtype=torch.bool but got dtype={mask.dtype} " \
            f"instead."
        assert mask.shape == self.size, \
            f"Expected mask of size {self.size} but got " \
            f"{mask.shape} instead."
        self._mask = mask

    def shoot(
            self, xyz, img_xyz, img_extrinsic, crop_top=0, crop_bottom=0,
            r_max=30, r_min=0.5):
        # We store indices in int64 format so we only accept indices up to
        # torch.iinfo(torch.long).max
        num_points = xyz.shape[0]
        if num_points >= 9223372036854775807:
            raise OverflowError

        # Initialize the indices to keep track of selected points
        indices = torch.arange(num_points, device=xyz.device)

        # Remove points outside of image range
        dist = torch.linalg.norm(xyz - img_xyz, dim=1)
        in_range = torch.where(torch.logical_and(r_min < dist, dist < r_max))[0]
        xyz = xyz[in_range]
        dist = dist[in_range]
        indices = indices[in_range]

        # Actual camera projection
        x_proj, y_proj, z_proj = self.project(xyz, img_xyz, img_extrinsic)

        # Remove points outside of camera field of view
        in_fov = self.field_of_view(
            x_proj, y_proj, crop_top=crop_top, crop_bottom=crop_bottom,
            z=z_proj)
        dist = dist[in_fov]
        indices = indices[in_fov]
        x_proj = x_proj[in_fov]
        y_proj = y_proj[in_fov]

        return indices, dist, x_proj, y_proj

    def field_of_view(self, x_pix, y_pix, z=None, crop_top=0, crop_bottom=0):
        f = field_of_view_cuda if x_pix.is_cuda else field_of_view_cpu
        return f(
            x_pix, y_pix, x_min=0, x_max=self.size[0], y_min=crop_top,
            y_max=self.size[1] - crop_bottom, z=z, mask=self.mask)

    @abstractmethod
    def project(self, xyz, img_xyz, img_extrinsic):
        """Projection of points in the camera, given the camera pose.
        Returns u, v and d float pixel coordinates.

        This method must be overridden in child classes.
        """
        pass

    @abstractmethod
    def splat(
            self, x_proj, y_proj, dist, xyz, img_extrinsic, crop_top=0,
            crop_bottom=0, voxel=0.02, k_swell=1.0, d_swell=1000):
        """Computation of voxel splats for each point. Returns rect
        coordinates for each point

        This method must be overridden in child classes.
        """
        pass


# -------------------------------------------------------------------- #
#                        Equirectangular Cameras                       #
# -------------------------------------------------------------------- #

class EquirectangularCamera(Camera):
    """Camera model producing equirectangular images of the whole sphere
     surrounding the camera.

    Attributes
        size:tuple         size of the pixel grid
        mask:BoolTensor    boolean mask for point projection
    """

    def project(self, xyz, img_xyz, img_extrinsic):
        """Compute the projection of 3D points into the image pixel
        coordinate system of an equirectangular camera.
        """
        # Compute the distance between the points and the camera
        xyz_to_img = xyz - img_xyz
        radius = norm_cuda(xyz_to_img)

        f = self.project_cuda if xyz.is_cuda else self.project_cpu
        return f(
            xyz_to_img, radius, img_extrinsic, self.size)

    @staticmethod
    def project_cuda(xyz_to_img, radius, img_opk, img_size):
        # Rotation matrix from image Euler angle pose
        rotation = opk_to_rotation_matrix_cuda(img_opk)

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

        return w_pix, h_pix, None

    @staticmethod
    def project_cpu(xyz_to_img, radius, img_opk, img_size):
        # Rotation matrix from image Euler angle pose
        rotation = opk_to_rotation_matrix_cpu(img_opk)

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

        return w_pix, h_pix, None

    def splat(
            self, x_proj, y_proj, dist, xyz, img_extrinsic, crop_top=0,
            crop_bottom=0, voxel=0.02, k_swell=1.0, d_swell=1000):
        f = self.splat_cuda if x_proj.is_cuda else self.splat_cpu
        return f(
            x_proj, y_proj, dist, self.size, crop_top, crop_bottom,
            voxel, k_swell, d_swell)

    @staticmethod
    def splat_cuda(
            x_proj, y_proj, dist, img_size, crop_top, crop_bottom,
            voxel, k_swell, d_swell):
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

    @staticmethod
    @njit(cache=True, nogil=True)
    def splat_cpu(
            x_proj, y_proj, dist, img_size, crop_top, crop_bottom,
            voxel, k_swell, d_swell):
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


# -------------------------------------------------------------------- #
#                          Perspective Cameras                         #
# -------------------------------------------------------------------- #

class PerspectiveCamera(Camera):
    """Perspective (or pinhole) camera model.

    Note: the default intrinsic parameters are those of KITTI360's
    perspective camera. Make sure you edit those to match your own
    camera.

    Attributes
        size:tuple          size of the pixel grid
        fx                  focal length x
        fy                  focal length y
        mx                  principal point offset x
        my                  principal point offset x
        mask:BoolTensor     boolean mask for point projection
    """
    _INVERT = False

    def __init__(
            self, size=(1408, 376), fx=788.629315, fy=786.38223, mx=687.158398,
            my=317.752196, mask=None):
        super().__init__(size=size, mask=mask)
        self.fx = fx
        self.fy = fy
        self.mx = mx
        self.my = my

    def project(self, xyz, img_xyz, img_extrinsic):
        """Compute the projection of 3D points into the image pixel
        coordinate system of a pinhole camera localized by a 4x4
        extrinsic parameters tensors.
        """
        f = self.project_cuda if xyz.is_cuda else self.project_cpu
        return f(
            xyz, img_extrinsic, self.fx, self.fy, self.mx, self.my,
            self._INVERT)

    @staticmethod
    def project_cuda(xyz, img_extrinsic, fx, fy, mx, my, invert):
        if invert:
            camera_to_world = torch.inverse(img_extrinsic)
            T = camera_to_world[:3, 3].view(3, 1)
            R = camera_to_world[:3, :3]
            p = R @ xyz.T + T
        else:
            camera_to_world = img_extrinsic
            T = camera_to_world[:3, 3].view(1, 3)
            R = camera_to_world[:3, :3]
            p = R.T @ (xyz - T).T

        p[0] = p[0] * fx / p[2] + mx
        p[1] = p[1] * fy / p[2] + my

        return p[0], p[1], p[2]

    @staticmethod
    @njit(cache=True, nogil=True)
    def project_cpu(xyz, img_extrinsic, fx, fy, mx, my, invert):
        # Recover the 4x4 camera-to-world matrix
        if invert:
            camera_to_world = np.linalg.inv(img_extrinsic)
            T = camera_to_world[:3, 3].copy().reshape((3, 1))
            R = camera_to_world[:3, :3].copy()
            p = R @ xyz.T + T
        else:
            camera_to_world = img_extrinsic
            T = camera_to_world[:3, 3].copy().reshape((1, 3))
            R = camera_to_world[:3, :3].copy()
            p = R.T @ (xyz - T).T

        p[0] = p[0] * fx / p[2] + mx
        p[1] = p[1] * fy / p[2] + my

        return p[0], p[1], p[2]

    def splat(
            self, x_proj, y_proj, dist, xyz, img_extrinsic, crop_top=0,
            crop_bottom=0, voxel=0.02, k_swell=1.0, d_swell=1000):
        f = self.splat_cuda if x_proj.is_cuda else self.splat_cpu
        return f(
            x_proj, y_proj, dist, self.fx, self.fy, self.size, crop_top,
            crop_bottom, voxel, k_swell, d_swell)

    @staticmethod
    def splat_cuda(
            x_proj, y_proj, dist, fx, fy, img_size, crop_top, crop_bottom,
            voxel, k_swell, d_swell):
        # Compute angular width. 3D points' projected masks are grown based
        # on their distance. Close-by points are further swollen with a
        # heuristic based on k_swell and d_swell.
        # Small angular widths assumption: tan(x)~x
        swell = (1 + k_swell * torch.exp(-dist / np.log(d_swell))) * voxel / dist
        width_x = swell * fx
        width_y = swell * fy
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

    @staticmethod
    @njit(cache=True, nogil=True)
    def splat_cpu(
            x_proj, y_proj, dist, fx, fy, img_size, crop_top, crop_bottom,
            voxel, k_swell, d_swell):
        # Compute angular width. 3D points' projected masks are grown based
        # on their distance. Close-by points are further swollen with a
        # heuristic based on k_swell and d_swell.
        # Small angular widths assumption: tan(x)~x
        swell = (1 + k_swell * np.exp(-dist / np.log(d_swell))) * voxel / dist
        width_x = swell * fx
        width_y = swell * fy

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


class KITTI360PerspectiveCamera(PerspectiveCamera):
    """Perspective (or pinhole) camera model for KITTI360 dataset.

    Attributes
        size:tuple          size of the pixel grid
        fx                  focal length x
        fy                  focal length y
        mx                  principal point offset x
        my                  principal point offset x
        mask:BoolTensor     boolean mask for point projection
    """
    _INVERT = False


class ScanNetCamera(PerspectiveCamera):
    """Perspective (or pinhole) camera modelfor ScanNet dataset.

    Attributes
        size:tuple          size of the pixel grid
        fx                  focal length x
        fy                  focal length y
        mx                  principal point offset x
        my                  principal point offset x
        mask:BoolTensor     boolean mask for point projection
    """
    _INVERT = True


# -------------------------------------------------------------------- #
#                            Fisheye Cameras                           #
# -------------------------------------------------------------------- #

class KITTI360Fisheye(Camera):
    """Fisheye camera model for KITTI360 dataset.

    Note: the default intrinsic parameters are those of KITTI360's
    fisheye camera number 2. Make sure you edit those to match your own
    camera.

    Attributes
        size:tuple             size of the pixel grid
        xi
        k1
        k2
        gamma1
        gamma2
        u0
        v0
        mask:BoolTensor        boolean mask for point projection
    """

    def __init__(
            self, size=(1400, 1400), xi=2.2134047507854890,
            k1=1.6798235660113681e-02, k2=1.6548773243373522e+00,
            gamma1=1.3363220825849971e+03, gamma2=1.3357883350012958e+03,
            u0=7.1694323510126321e+02, v0=7.0576498308221585e+02, mask=None):
        super().__init__(size=size, mask=mask)
        self.xi = xi
        self.k1 = k1
        self.k2 = k2
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.u0 = u0
        self.v0 = v0

    def project(self, xyz, img_xyz, img_extrinsic):
        """Compute the projection of 3D points into the image pixel
        coordinate system of a fisheye camera localized by a 4x4
        extrinsic parameters tensors.

        Credit: https://github.com/autonomousvision/kitti360Scripts
        """
        f = self.project_cuda if xyz.is_cuda else self.project_cpu
        return f(
            xyz, img_extrinsic, self.xi, self.k1, self.k2, self.gamma1,
            self.gamma2, self.u0, self.v0)

    @staticmethod
    def project_cuda(xyz, img_extrinsic, xi, k1, k2, gamma1, gamma2, u0, v0):
        # Place the points in the camera's coordinate system
        camera_to_world = img_extrinsic
        T = camera_to_world[:3, 3].view(1, 3)
        R = camera_to_world[:3, :3]
        p = R.T @ (xyz - T).T

        # Compute float pixel coordinates
        p = p.T
        norm = torch.linalg.norm(p, dim=1)

        x = p[:, 0] / norm
        y = p[:, 1] / norm
        z = p[:, 2] / norm

        x /= z + xi
        y /= z + xi

        r2 = x ** 2 + y ** 2
        r4 = r2 ** 2

        x = gamma1 * (1 + k1 * r2 + k2 * r4) * x + u0
        y = gamma2 * (1 + k1 * r2 + k2 * r4) * y + v0

        return x, y, norm * p[:, 2] / (p[:, 2].abs() + 1e-4)

    @staticmethod
    @njit(cache=True, nogil=True)
    def project_cpu(xyz, img_extrinsic, xi, k1, k2, gamma1, gamma2, u0, v0):
        # Place the points in the camera's coordinate system
        camera_to_world = img_extrinsic
        T = camera_to_world[:3, 3].copy().reshape((1, 3))
        R = camera_to_world[:3, :3].copy()
        p = R.T @ (xyz - T).T

        # Compute float pixel coordinates
        p = p.T
        norm = np.linalg.norm(p, axis=1)

        x = p[:, 0] / norm
        y = p[:, 1] / norm
        z = p[:, 2] / norm

        x /= z + xi
        y /= z + xi

        r2 = x ** 2 + y ** 2
        r4 = r2 ** 2

        x = gamma1 * (1 + k1 * r2 + k2 * r4) * x + u0
        y = gamma2 * (1 + k1 * r2 + k2 * r4) * y + v0

        return x, y, norm * p[:, 2] / np.abs(p[:, 2] + 1e-4)

    def splat(
            self, x_proj, y_proj, dist, xyz, img_extrinsic, crop_top=0,
            crop_bottom=0, voxel=0.02, k_swell=1.0, d_swell=1000):
        f = self.splat_cuda if x_proj.is_cuda else self.splat_cpu
        return f(
            x_proj, y_proj, xyz, img_extrinsic, self.xi, self.k1, self.k2,
            self.gamma1, self.gamma2, self.u0, self.v0, self.size, crop_top,
            crop_bottom, voxel, k_swell, d_swell)

    @staticmethod
    def splat_cuda(
            x_proj, y_proj, xyz, img_extrinsic, xi, k1, k2, gamma1, gamma2, u0,
            v0, img_size, crop_top, crop_bottom, voxel, k_swell, d_swell):
        # Compute angular width. 3D points' projected masks are grown based
        # on their distance. Close-by points are further swollen with a
        # heuristic based on k_swell and d_swell.
        # Small angular widths assumption: tan(x)~x
        dist = norm_cuda(xyz)
        swell = (1 + k_swell * torch.exp(-dist / np.log(d_swell)))

        # Compute the projection of the top of the voxel / cube. The
        # distance between this projection and the actual point's projection
        # will serve as a proxy to estimate the rect's size. This heuristic
        # does not hold if voxels are quite large and close to the camera,
        # this should not cause too much trouble for outdoor scenes but may
        # affect narrow indoor scenes with close-by objects such as walls
        # TODO: improve fisheye rect computation
        z_offset = torch.zeros_like(xyz)
        z_offset[:, 2] = swell * voxel / 2
        x, y, _ = KITTI360Fisheye.project_cuda(
            xyz + z_offset, img_extrinsic, xi, k1, k2, gamma1, gamma2, u0, v0)
        splat_xy_width = 2 * (
                (x_proj - x) ** 2 + (y_proj - y) ** 2).sqrt().repeat(2, 1).t()

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

    @staticmethod
    @njit(cache=True, nogil=True)
    def splat_cpu(
            x_proj, y_proj, xyz, img_extrinsic, xi, k1, k2, gamma1, gamma2, u0,
            v0, img_size, crop_top, crop_bottom, voxel, k_swell, d_swell):
        # Compute angular width. 3D points' projected masks are grown based
        # on their distance. Close-by points are further swollen with a
        # heuristic based on k_swell and d_swell.
        # Small angular widths assumption: tan(x)~x
        dist = norm_cpu(xyz)
        swell = (1 + k_swell * np.exp(-dist / np.log(d_swell)))

        # Compute the projection of the top of the voxel / cube. The
        # distance between this projection and the actual point's projection
        # will serve as a proxy to estimate the rect's size. This heuristic
        # does not hold if voxels are quite large and close to the camera,
        # this should not cause too much trouble for outdoor scenes but may
        # affect narrow indoor scenes with close-by objects such as walls
        # TODO: improve fisheye rect computation
        z_offset = np.zeros_like(xyz)
        z_offset[:, 2] = swell * voxel / 2
        x, y, _ = KITTI360Fisheye.project_cpu(
            xyz + z_offset, img_extrinsic, xi, k1, k2, gamma1, gamma2, u0, v0)
        splat_xy_width = 2 * np.tile(
            np.sqrt((x_proj - x) ** 2 + (y_proj - y) ** 2), (2, 1)).T

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
