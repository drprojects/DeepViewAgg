import copy
import numpy as np
from PIL import Image
import torch
import torch_scatter
from typing import List
from tqdm.auto import tqdm as tq
from torch_points3d.core.multimodal import CSRData, CSRBatch
from torch_points3d.utils.multimodal import lexargsort, lexunique, \
    lexargunique, CompositeTensor
from torch_points3d.utils.multimodal import tensor_idx
from torch_points3d.core.multimodal.camera import opk_to_rotation_matrix_cuda
from torch_points3d.core.multimodal.camera import Camera


# -------------------------------------------------------------------- #
#                           Camera Intrinsic                           #
# -------------------------------------------------------------------- #

def _adjust_intrinsic(
        in_fx, in_fy, in_mx, in_my, in_size, out_size=None, offset=None):
    """Adjust fx, fy, mx and my intrinsic parameters after an image
    resizing or cropping.

    Inspired from: https://github.com/angeladai/3DMV
    """
    n = in_fx.shape[0]
    has_offset = offset is not None
    has_resize = out_size is not None and in_size != out_size
    assert all([x.shape[0] == n] for x in [in_fy, in_mx, in_my])
    assert not has_offset or offset.shape == (n, 2)

    # Prepare output intrinsic parameters
    fx = in_fx
    fy = in_fy
    mx = in_mx
    my = in_my

    # Adapt focal lengths after resizing
    if has_resize:
        resize_width = int(np.floor(
            out_size[1] * float(in_size[0]) / float(in_size[1])))
        fx *= float(resize_width) / float(in_size[0])
        fy *= float(out_size[1]) / float(in_size[1])

    # Adapt principal point offset after offsetting (offsetting
    # assumed to be expressed in the in_size coordinate system)
    if has_offset:
        mx -= offset[:, 0]
        my -= offset[:, 1]

    if has_resize:
        mx *= float(out_size[0] - 1) / float(in_size[0] - 1)
        my *= float(out_size[1] - 1) / float(in_size[1] - 1)

    return fx, fy, mx, my


def adjust_intrinsic(func):
    def wrapper(self, *args, **kwargs):
        # assert isinstance(self, SameSettingImageData)

        if not self.is_pinhole:
            return func(self, *args, **kwargs)

        # Try-except to handle the edge-case where SameSettingImageData
        # attributes are not all set. This happens in the __init__
        # constructor when first setting important attributes such as
        # 'cam_size', 'crop_offsets' without which the
        # intrinsic parameters cannot be re-computed
        try:
            # Gather image parameters before func
            in_fx = self.fx
            in_fy = self.fy
            in_mx = self.mx
            in_my = self.my
            in_size = self.cam_size
            in_offsets = self.crop_offsets

            out = func(self, *args, **kwargs)

            # Gather image parameters after func
            # TODO: take crop_size into account in projection ?
            out_size = self.cam_size
            out_offsets = self.crop_offsets

            # Adjust intrinsic parameters to take resizing and cropping
            # into account
            self.fx, self.fy, self.mx, self.my = _adjust_intrinsic(
                in_fx, in_fy, in_mx, in_my, in_size, out_size=out_size,
                offset=out_offsets - in_offsets)
        except:
            out = func(self, *args, **kwargs)

        return out

    return wrapper


# -------------------------------------------------------------------- #
#                      Sparse Image Interpolation                      #
# -------------------------------------------------------------------- #

def sparse_interpolation(features, coords, batch, padding_mode='border'):
    """Interpolate a batch of feature maps of size (B, C, H, W) only at
    given pixel coordinates. This function is equivalent to
    `torch.nn.functional.grid_sample` with `mode='bilinear'`,
    `padding_mode='zeros'` and `align_corners=False`, but allows queried
    pixel coordinates to be different for each feature map.

    :param features: feature map of size (B, C, H, W)
    :param coords: tensor of size (N, 2) holding float interpolation
      coordinates in [0, 1]. To convert interpolation pixel coordinates
      to such float coordinates, use:
      `pixel_coordinate / (output_resolution - 1)`
    :param batch: LongTensor of size (N) indicating, for each row of
      `coords`, which feature map should be interpolated
    :param padding_mode: string specifying the padding mode for outside
      grid values ``'zeros'`` | ``'border'`` | ``'reflection'``.
      Default: ``'border'``
    :return: tensor of size (N, C) of interpolated features
    """
    assert len(features.shape) == 4
    assert coords.shape[0] == batch.shape[0]
    assert len(coords.shape) == 2 and coords.shape[1] == 2
    assert 0 <= coords.min()
    assert coords.max() <= 1

    # Pad images with 0-feature
    if padding_mode == 'zeros':
        images_pad = torch.nn.ZeroPad2d(1)(features)
    elif padding_mode == 'border':
        images_pad = torch.nn.ReplicationPad2d(1)(features)
    elif padding_mode == 'reflection':
        images_pad = torch.nn.ReflectionPad2d(1)(features)
    else:
        raise NotImplementedError(f"Unknown padding_mode='{padding_mode}'")

    # Recover the image dimensions
    h, w = features.shape[2:]

    # Adapt [0, 1] coordinates to padded image coordinate system
    pixels = coords * torch.Tensor([[h, w]]).to(features.device) + 0.5

    # Compute the interpolation pixel coordinates: top-left, top-right,
    # bottom-left, bottom-right
    # NB: torch.ceil(x) != torch.floor(x+1) when x is an integer. So we
    # favor torch.floor(x+1) over torch.ceil here
    top = torch.floor(pixels[:, 0])
    bottom = torch.floor(pixels[:, 0] + 1)
    left = torch.floor(pixels[:, 1])
    right = torch.floor(pixels[:, 1] + 1)
    pixels_tl = torch.stack((top, left)).T.long()
    pixels_tr = torch.stack((top, right)).T.long()
    pixels_bl = torch.stack((bottom, left)).T.long()
    pixels_br = torch.stack((bottom, right)).T.long()

    # Compute the weight associated with each interpolation point
    w_tl = torch.prod(pixels - pixels_br, dim=1).abs().unsqueeze(1)
    w_tr = torch.prod(pixels - pixels_bl, dim=1).abs().unsqueeze(1)
    w_bl = torch.prod(pixels - pixels_tr, dim=1).abs().unsqueeze(1)
    w_br = torch.prod(pixels - pixels_tl, dim=1).abs().unsqueeze(1)

    out = w_tl * images_pad[batch, :, pixels_tl[:, 0], pixels_tl[:, 1]] \
          + w_tr * images_pad[batch, :, pixels_tr[:, 0], pixels_tr[:, 1]] \
          + w_bl * images_pad[batch, :, pixels_bl[:, 0], pixels_bl[:, 1]] \
          + w_br * images_pad[batch, :, pixels_br[:, 0], pixels_br[:, 1]]

    return out


# -------------------------------------------------------------------- #
#                              Image Data                              #
# -------------------------------------------------------------------- #

# TODO: cam_size is a scalar or a list of size num_cameras
# TODO: if no cameras; cam_size MUST be et and MUST be a scalar (list not supported)
# TODO: cam_size cannot be set if cameras exist
# TODO: if cam_size exists, cameras can only be added



# TODO: clean pass over the different image sizes

# TODO: SCALE and CROP : project on camera-based img_size (make sure
#  INTRINSICS are up to date) and CROP only after projection: the CAMERA
#  receives the crop bbox and returns the relevant projection

# TODO: how and when to adjust_intrinsics ?

# TODO: modify NonStaticMask to run on the CAMERAS

# TODO: clean all global calls to SameSettingImageData !!!



class SameSettingImageData:
    """Class to hold arrays of images information, along with shared
    3D-2D mapping information.

    Attributes
        path:numpy.ndarray          [N] paths
        pos:torch.Tensor            [Nx3] positions
        extrinsic:torch.Tensor      [Nx4x4] or [Nx3] extrinsic parameters. The latter format expects omega-phi-kappa angles

        cam_size:tuple              initial size of the loaded images and mappings
        downscale:float             downsampling of images and mappings wrt cam_size
        rollings:LongTensor         rolling offsets for each image wrt cam_size
        crop_size:tuple             size of the cropping box wrt cam_size
        crop_offsets:LongTensor     cropping box offsets for each image wrt cam_size

        x:Tensor                    tensor of features
        mappings:ImageMapping       mappings between 3D points and the images
    """
    _numpy_keys = ['path', 'cameras', 'camera_idx']
    _torch_keys = ['pos', 'x', 'extrinsic', 'crop_offsets', 'rollings']
    _map_key = 'mappings'
    _shared_keys = ['cam_size', 'downscale', 'crop_size', 'cameras']
    _own_keys = _numpy_keys + _torch_keys + [_map_key]
    _keys = _shared_keys + _own_keys

    def __init__(
            self, path=np.empty(0, dtype='O'), pos=torch.empty([0, 3]),
            extrinsic=torch.empty([0, 4, 4]), cam_size=(512, 256),
            downscale=1, rollings=None, crop_size=None, crop_offsets=None,
            x=None, mappings=None, cameras=None, camera_idx=None, **kwargs):

        assert path.shape[0] == pos.shape[0] == extrinsic.shape[0]

        self.cam_size = cam_size
        self._path = np.asarray(path)
        self._pos = pos.double()
        self._extrinsic = extrinsic.double()
        self.rollings = rollings if rollings is not None \
            else torch.zeros(self.num_views, dtype=torch.int64)
        self.crop_size = crop_size if crop_size is not None else self.cam_size
        self.crop_offsets = crop_offsets if crop_offsets is not None \
            else torch.zeros((self.num_views, 2), dtype=torch.int64)
        self.downscale = downscale
        self.x = x
        self.mappings = mappings
        self.cameras = cameras
        self.camera_idx = camera_idx

        # self.debug()

    def debug(self):
        assert self.path.shape[0] == self.num_views, \
            f"Attributes 'path' and 'pos' must have the same length."

        assert self.has_opk_extrinsic != self.has_4x4_extrinsic, \
            f"Poses must either be provided as Omega-Phi-Kappa angles or as " \
            f"a full 4x4 extrinsic matrix."

        assert self.device == self.extrinsic.device, \
            f"Discrepancy in the devices of 'pos' and 'extrinsic' " \
            f"attributes. Please use `SameSettingImageData.to()` to set " \
            f"the device."

        assert len(tuple(self.cam_size)) == 2, \
            f"Expected len(cam_size)=2 but got {len(self.cam_size)} instead."
        assert self.rollings.shape[0] == self.num_views, \
            f"Expected tensor of size {self.num_views} but got " \
            f"{self.rollings.shape[0]} instead."
        assert len(tuple(self.crop_size)) == 2, \
            f"Expected len(crop_size)=2 but got {len(self.crop_size)} instead."
        assert all(a <= b for a, b in zip(self.crop_size, self.cam_size)), \
            f"Expected size smaller than {self.cam_size} but got " \
            f"{self.crop_size} instead."
        assert self.crop_offsets.shape == (self.num_views, 2), \
            f"Expected tensor of shape {(self.num_views, 2)} but got " \
            f"{self.crop_offsets.shape} instead."
        assert self._downscale >= 1, \
            f"Expected scalar larger than 1 but got {self._downscale} instead."

        if self.x is not None:
            assert isinstance(self.x, torch.Tensor), \
                f"Expected a tensor of image features but got " \
                f"{type(self.x)} instead."
            assert self.x.shape[0] == self.num_views \
                   and self.x.shape[2] == self.img_size[1] \
                   and self.x.shape[3] == self.img_size[0], \
                f"Expected a tensor of shape ({self.num_views}, :, " \
                f"{self.img_size[1]}, {self.img_size[0]}) but got " \
                f"{self.x.shape} instead."
            assert self.device == self.x.device, \
                f"Discrepancy in the devices of 'pos' and 'x' attributes. " \
                f"Please use `SameSettingImageData.to()` to set the device."

        if self.mappings is not None:
            assert isinstance(self.mappings, ImageMapping), \
                f"Expected an ImageMapping but got {type(self.mappings)} " \
                f"instead."
            unique_idx = torch.unique(self.mappings.images)
            img_idx = torch.arange(self.num_views, device=self.device)
            assert (unique_idx == img_idx).all(), \
                f"Image indices in the mappings do not match the " \
                f"SameSettingImageData image indices."
            if self.mappings.num_items > 0:
                w_max, h_max = self.mappings.pixels.max(dim=0).values
                assert w_max < self.crop_size[0] and h_max < self.crop_size[1], \
                    f"Max pixel values should be smaller than ({self.crop_size}) " \
                    f"but got ({w_max, h_max}) instead."
            assert self.device == self.mappings.device, \
                f"Discrepancy in the devices of 'pos' and 'mappings' " \
                f"attributes. Please use `SameSettingImageData.to()` to set " \
                f"the device."
            self.mappings.debug()

    def to_dict(self):
        return {key: getattr(self, key) for key in self._keys}

    @property
    def path(self):
        return self._path

    @property
    def pos(self):
        return self._pos

    @property
    def extrinsic(self):
        return self._extrinsic

    @property
    def num_views(self):
        return self.pos.shape[0]

    @property
    def has_4x4_extrinsic(self):
        return self.extrinsic.shape == (self.num_views, 4, 4)

    @property
    def has_opk_extrinsic(self):
        return self.extrinsic.shape == (self.num_views, 3)

    @property
    def axes(self):
        if self.has_opk_extrinsic:
            rotations = torch.cat([
                opk_to_rotation_matrix_cuda(x).unsqueeze(0)
                for x in self.extrinsic.view(-1, 3)], dim=0)
        elif self.has_4x4_extrinsic:
            rotations = self.extrinsic[:, :3, :3].transpose(1, 2)
        else:
            raise ValueError('No available pose information to compute axes.')
        return rotations

    @property
    def num_points(self):
        """Number of points implied by ImageMapping. Zero is 'mappings'
        is None.
        """
        return self.mappings.num_groups if self.mappings is not None else 0

    @property
    def img_size(self):
        """Current size of the 'x' and 'mappings'. Depends on the
        cropping size and the downsampling scale.
        """
        return tuple(int(x / self.downscale) for x in self.crop_size)

    @property
    def cam_size(self):
        """Initial size of the loaded image features and the mappings.

        This size is used as reference to characterize other
        SameSettingImageData attributes such as the crop offsets,
        resolution. As such, it should not be modified directly.
        """
        return self._cam_size

    @cam_size.setter
    @adjust_intrinsic
    def cam_size(self, cam_size):
        cam_size = tuple(cam_size)
        if self.cam_size == cam_size:
            return
        assert self.x is None and self.mappings is None, \
            "Can't edit 'cam_size' if 'x', 'mappings' and 'cameras' are not all None."
        assert len(cam_size) == 2, \
            f"Expected len(cam_size)=2 but got {len(cam_size)} instead."
        # Warning: modifying 'cam_size', has the effect of resetting 'crop_size'
        self._cam_size = cam_size
        self.crop_size = cam_size

    @property
    def pixel_dtype(self):
        """Smallest torch dtype allowed by the resolution for encoding
        pixel coordinates.
        """
        size = self.cam_size
        if size is None:
            return torch.int64
        for dtype in [torch.int16, torch.int32, torch.int64]:
            if torch.iinfo(dtype).max >= max(size[0], size[1]):
                break
        return dtype

    @property
    def rollings(self):
        """Rollings to apply to each image, with respect to the
        'cam_size' state.

        By convention, rolling is applied first, then cropping, then
        resizing. For that reason, rollings should be defined before
        'x' or 'mappings' are cropped or resized.
        """
        return getattr(self, '_rollings', None)

    @rollings.setter
    def rollings(self, rollings):
        assert (self.x is None and self.mappings is None) \
               or (self.rollings == rollings).all(), \
            "Can't directly edit 'rollings' if 'x' or 'mappings' are " \
            "not both None. Consider using 'update_rollings'."
        assert rollings.dtype == torch.int64, \
            f"Expected dtype=torch.int64 but got dtype={rollings.dtype} " \
            f"instead."
        assert rollings.shape[0] == self.num_views, \
            f"Expected tensor of size {self.num_views} but got " \
            f"{rollings.shape[0]} instead."
        self._rollings = rollings.to(self.device)

    def update_rollings(self, rollings):
        """Update the rollings state of the SameSettingImageData, WITH
        RESPECT TO ITS REFERENCE STATE 'cam_size'.

        This assumes the images have a circular representation (ie that
        the first and last pixels along the width are adjacent in
        reality).

        Does not support prior cropping along the width or resizing.
        """
        # Make sure no prior cropping or resizing was applied to the
        # images and mappings
        assert self.cam_size[0] == self.img_size[0], \
            f"CenterRoll cannot operate if images and mappings " \
            f"underwent prior cropping or resizing."
        assert self.crop_size is None \
               or self.crop_size == self.cam_size, \
            f"CenterRoll cannot operate if images and mappings " \
            f"underwent prior cropping or resizing."
        assert self.downscale is None or self.downscale == 1, \
            f"CenterRoll cannot operate if images and mappings " \
            f"underwent prior cropping or resizing."

        # Edit the internal rollings attribute
        self._rollings = rollings

        # Roll the image features
        if self.x is not None:
            x = [torch.roll(im, roll.item(), dims=-1)
                 for im, roll in zip(self.x, self.rollings)]
            x = torch.cat([im.unsqueeze(0) for im in x])
            self.x = x

        # Roll the mappings
        if self.mappings is not None:
            # Expand the rollings
            pix_roll = self.rollings[self.mappings.images].repeat_interleave(
                self.mappings.values[1].pointers[1:]
                - self.mappings.values[1].pointers[:-1])

            # Recover the width pixel coordinates
            w_pix = self.mappings.pixels[:, 0].long()
            w_pix = (w_pix + pix_roll) % self.cam_size[0]
            w_pix = w_pix.type(self.pixel_dtype)

            # Apply pixel update
            self.mappings.pixels[:, 0] = w_pix

        # TODO: Roll the mask, intrinsics and extrinsics

        return self

    @property
    def crop_size(self):
        """Size of the cropping to apply to the 'cam_size' to obtain the
        current image cropping.

        This size is used to characterize 'x' and 'mappings'. As
        such, it should not be modified directly.
        """
        return getattr(self, '_crop_size', None)

    @crop_size.setter
    def crop_size(self, crop_size):
        crop_size = tuple(crop_size)
        if self.crop_size == crop_size:
            return
        assert (self.x is None and self.mappings is None), \
            "Can't directly edit 'crop_size' if 'x' or 'mappings' are " \
            "not both None. Consider using 'update_cropping'."
        assert len(crop_size) == 2, \
            f"Expected len(crop_size)=2 but got {len(crop_size)} instead."
        assert crop_size[0] <= self.cam_size[0] \
               and crop_size[1] <= self.cam_size[1], \
            f"Expected size smaller than {self.cam_size} but got " \
            f"{crop_size} instead."
        self._crop_size = crop_size

    @property
    def mapping_size(self):
        """Image size for the mappings."""
        return self.crop_size

    @property
    def crop_offsets(self):
        """X-Y (width, height) offsets of the top-left corners of
        cropping boxes to apply to the 'cam_size' in order to obtain the
        current image cropping.

        These offsets must match the 'num_views' and is used to
        characterize 'x' and 'mappings'. As such, it should not be
        modified directly.
        """
        return getattr(self, '_crop_offsets', None)

    @crop_offsets.setter
    @adjust_intrinsic
    def crop_offsets(self, crop_offsets):
        assert (self.x is None and self.mappings is None) \
               or (self.crop_offsets == crop_offsets).all(), \
            "Can't directly edit 'crop_offsets' if 'x' or 'mappings' are not " \
            "both None. Consider using 'update_cropping'."
        assert crop_offsets.dtype == torch.int64, \
            f"Expected dtype=torch.int64 but got dtype={crop_offsets.dtype} " \
            f"instead."
        assert crop_offsets.shape == (self.num_views, 2), \
            f"Expected tensor of shape {(self.num_views, 2)} but got " \
            f"{crop_offsets.shape} instead."
        self._crop_offsets = crop_offsets.to(self.device)

    def update_cropping(self, crop_size, crop_offsets):
        """Update the cropping state of the SameSettingImageData, WITH
        RESPECT TO ITS CURRENT STATE 'img_size'.

        Parameters crop_size and crop_offsets are resized to the
        'cam_size'

        Crop the 'x' and 'mappings', with respect to their current
        'img_size' (as opposed to the 'cam_size').
        """
        # Update the private 'crop_size' and 'crop_offsets' attributes
        # wrt 'cam_size'
        crop_offsets = crop_offsets.long()
        self._crop_size = tuple(int(x * self.downscale) for x in crop_size)
        self._crop_offsets = (
                self.crop_offsets + crop_offsets * self.downscale).long()

        # Update the images' cropping
        #   - Image features have format: BxCxHxW
        #   - Crop size has format: (W, H)
        #   - Crop offsets have format: (W, H)
        if self.x is not None:
            x = [
                im[:, o[1]:o[1] + crop_size[1], o[0]:o[0] + crop_size[0]]
                for im, o in zip(self.x, crop_offsets)]
            x = torch.cat([im.unsqueeze(0) for im in x])
            self.x = x

        # Update the mappings
        if self.mappings is not None:
            self.mappings = self.mappings.crop(crop_size, crop_offsets)

        return self

    @property
    def downscale(self):
        """Downsampling scale factor of the current image resolution,
        with respect to the initial image size 'cam_size'.

        Must follow: scale >= 1
        """
        return getattr(self, '_downscale', None)

    @downscale.setter
    def downscale(self, scale):
        assert (self.x is None and self.mappings is None) \
               or self.downscale == scale, \
            "Can't directly edit 'downscale' if 'x' or 'mappings' are " \
            "not both None. Setting 'x' will automatically adjust the " \
            "scale."
        assert scale >= 1, \
            f"Expected scalar larger than 1 but got {scale} instead."
        # assert isinstance(scale, int), \
        #     f"Expected an int but got a {type(scale)} instead."
        # assert (scale & (scale-1) == 0) and scale != 0,\
        #     f"Expected a power of 2 but got {scale} instead."
        self._downscale = scale

    @property
    def x(self):
        """Tensor of loaded image features with shape NxCxHxW, where
        N='num_views' and (W, H)='img_size'. Can be None if no image
        features were loaded.

        For clean load, consider using 'SameSettingImageData.load()'.
        """
        return getattr(self, '_x', None)

    @x.setter
    def x(self, x):
        if x is None:
            self._x = None
            return

        assert isinstance(x, torch.Tensor), \
            f"Expected a tensor of image features but got {type(x)} " \
            f"instead."
        assert x.shape[0] == self.num_views, \
            f"Expected a tensor of shape ({self.num_views}, :, " \
            f"{self.img_size[1]}, {self.img_size[0]}) but got " \
            f"{x.shape} instead."
        # TODO: removing this constraint as it may be broken when down
        # assert x.shape[2:][::-1] == self.img_size, \
        #     f"Expected a tensor of shape ({self.num_views}, :, " \
        #     f"{self.img_size[1]}, {self.img_size[0]}) but got " \
        #     f"{x.shape} instead."

        # TODO: treat scales independently. Careful with min or max
        #  depending on upscale and downlscale
        # Update internal attributes based on the input downscaled image
        # features. We assume the scale hase been changed homogeneously
        # on both width and height, but this could be wrong for some
        # special cases with down convolutions. For security, since we
        # use only a single scalar to describe scale, we use the largest
        # rescaling so that mappings do not go out of frame.
        scale_x = self.img_size[0] / x.shape[3]
        scale_y = self.img_size[1] / x.shape[2]
        scale = max(scale_x, scale_y)
        self._downscale = self.downscale * scale
        self._x = x.to(self.device)

    @property
    def mappings(self):
        """ImageMapping data mapping 3D points to the images.

        The state of the mappings is closely linked to the state of the
        images. The image indices must agree with 'num_views', the
        pixel coordinates must correspond to the current 'img_size',
        scaling and cropping. As such, it is recommended not to
        manipulate the mappings directly.
        """
        return getattr(self, '_mappings', None)

    @mappings.setter
    def mappings(self, mappings):
        if mappings is None:
            self._mappings = None
            return

        assert isinstance(mappings, ImageMapping), \
            f"Expected an ImageMapping but got {type(mappings)} instead."
        # TODO: these calls to torch.unique and torch.Tensor.max()
        #  are particularly computation-intensive. They tend to slow
        #  down item selection for large datasets such as S3DIS.
        #  For now, I choose to prioritize speed over these
        #  sanity-checks.
        # unique_idx = torch.unique(mappings.images)
        # img_idx = torch.arange(self.num_views, device=self.device)
        # assert (unique_idx == img_idx).all(), \
        #     f"Image indices in the mappings do not match the " \
        #     f"SameSettingImageData image indices."
        # if mappings.num_items > 0:
        #     w_max, h_max = mappings.pixels.max(dim=0).values
        #     assert w_max < self.img_size[0] and h_max < self.img_size[1], \
        #         f"Max pixel values should be smaller than ({self.img_size}) " \
        #         f"but got ({w_max, h_max}) instead."
        self._mappings = mappings.to(self.device)

    @property
    def cameras(self):
        return getattr(self, '_cameras', None)

    @cameras.setter
    def cameras(self, cameras):
        assert isinstance(cameras, (list, np.ndarray))
        assert all(isinstance(x, Camera) for x in cameras)
        self._cameras = np.asarray(cameras)

    @property
    def num_cameras(self):
        return len(self.cameras) if self.cameras else 0

    @property
    def camera_idx(self):
        return getattr(self, '_camera_idx', None)

    @camera_idx.setter
    def camera_idx(self, camera_idx):
        assert self.cameras is not None
        assert isinstance(camera_idx, torch.LongTensor)
        assert camera_idx.dim() == 1
        assert camera_idx.shape[0] == self.num_views
        assert camera_idx.max() + 1 <= self.num_cameras
        self._camera_idx = camera_idx

    @property
    def camera_one_hot(self):
        if self.cameras is None:
            return None
        return torch.nn.functional.one_hot(
            self.camera_idx, num_classes=self.num_cameras)

    def select_points(self, idx, mode='pick'):
        """Update the 3D points sampling. Typically called after a 3D
        sampling or strided convolution layer in a 3D CNN encoder. For
        mappings to preserve their meaning, the corresponding 3D points
        are assumed to have been sampled with the same index.

        To update the 3D resolution in the modality data and mappings,
        two methods - ruled by the `mode` parameter - may be used:
        picking or merging.

          - 'pick' (default): only a subset of points is sampled, the
            rest is discarded. In this case, a 1D indexing array must be
            provided.

          - 'merge': points are agglomerated. The mappings are combined,
            duplicates are removed. If any other value (such as
            mapping features) is present in the mapping, the value
            of one of the duplicates is picked at random. In this case,
            the correspondence map for the N points in the mapping must
            be provided as a 1D array of size N such that i -> idx[i].

        Returns a new SameSettingImageData object.
        """
        # TODO: make sure the merge mode works on real data...

        # Convert idx to a convenient indexing format
        idx = tensor_idx(idx).to(self.device)

        # Work on a clone of self, to avoid in-place modifications.
        # Images are not affected if no mappings are present or idx is
        # None
        if self.mappings is None or idx is None or idx.shape[0] == 0:
            return self.clone()

        # If no images, still need to preserve the number of points in
        # the mappings
        if len(self) == 0:
            images = self.clone()
            # TODO: find out why this sometimes crashes and fix it
            # images.mappings.pointers = torch.zeros(
            #     idx.shape[0] + 1, dtype=torch.long, device=self.device)
            return images

        # Picking mode by default
        if mode == 'pick':
            # Select mappings wrt the point index
            mappings = self.mappings.select_points(idx, mode=mode)

            # Select the images used in the mappings. Selected images
            # are sorted by their order in image_indices. Mappings'
            # image indices will also be updated to the new ones.
            # Mappings are temporarily separating from self as they
            # will be affected by the indexing on images.
            seen_image_idx = lexunique(mappings.images) \
                if mappings.num_items > 0 else []
            self_mappings = self.mappings
            self.mappings = None
            images = self[seen_image_idx]
            images.mappings = mappings.select_images(seen_image_idx)
            self.mappings = self_mappings

        # Merge mode
        elif mode == 'merge':
            try:
                assert idx.shape[0] == self.num_points > 0, \
                    f"Merge correspondences has size {idx.shape[0]} but size " \
                    f"{self.num_points} was expected."
                assert (torch.arange(idx.max() + 1, device=self.device)
                        == torch.unique(idx)).all(), \
                    "Merge correspondences must map to a compact set of " \
                    "indices."
            except:
                # TODO: quick fix because we don't know why this occasionally crashes
                return self.clone()

            # Select mappings wrt the point index
            # Images are not modified, since the 'merge' mode
            # guarantees no image is discarded
            images = self.clone()
            images.mappings = images.mappings.select_points(idx, mode=mode)

        else:
            raise ValueError(f"Unknown point selection mode '{mode}'.")

        return images

    def select_views(self, view_mask):
        """Select the views. Typically called when selecting views based
        on their mapping features. So as to preserve the views ordering,
        view_mask is assumed to be a boolean mask over views.

        The mappings are updated so as to remove views to images absent
        from the view_mask and change the image indexing to respect the
        new order.

        To update the views in the modality data and mappings, the image
        may be subselected if any is absent from the selected mappings.

        Returns a new SameSettingImageData object.
        """

        # Images are not affected if no mappings are present or
        # view_mask is None or all True
        if self.mappings is None or view_mask is None or torch.all(view_mask) \
                or len(self) == 0:
            return self.clone()

        # Select mappings wrt the point index
        mappings, seen_image_idx = self.mappings.select_views(view_mask)

        # Select the images used in the mappings. Selected images
        # are sorted by their order in image_indices. Mappings'
        # image indices will also be updated to the new ones.
        # Mappings are temporarily removed from the images as they
        # will be affected by the indexing on images.
        if seen_image_idx is not None:
            self_mappings = self.mappings
            self.mappings = None
            images = self[seen_image_idx]
            self.mappings = self_mappings
        else:
            self_mappings = self.mappings
            self.mappings = None
            images = self.clone()
            self.mappings = self_mappings
        images.mappings = mappings

        return images

    def load(self, show_progress=False):
        """Load images to the 'x' attribute.

        Images are batched into a tensor of size NxCxHxW, where
        N='num_views' and (W, H)='img_size'. They are read with
        respect to their order in 'path', resized to 'cam_size', rolled
        with 'rollings', cropped with 'crop_size' and 'crop_offsets'
        and subsampled by 'downscale'.
        """
        self._x = self.read_images(
            size=self.cam_size,
            rollings=self.rollings,
            crop_size=self.crop_size,
            crop_offsets=self.crop_offsets,
            downscale=self.downscale,
            show_progress=show_progress).to(self.device)
        return self

    def read_images(
            self, idx=None, size=None, rollings=None, crop_size=None,
            crop_offsets=None, downscale=None, show_progress=False):
        # TODO: faster read with multiprocessing:
        #  https://stackoverflow.com/questions/19695249/load-just-part-of-an-image-in-python
        #  https://towardsdatascience.com/10x-faster-parallel-python-without-python-multiprocessing-e5017c93cce1
        """
        Read images and batch them into a tensor of size BxCxHxW.

        Images are indexed with 'idx' with respect to their order in
        'path', then resized to 'size', then rolled with 'rollings',
        before being cropped with 'crop_size' and 'crop_offsets' and
        subsampled by 'downscale'.
        """
        # Index to select part of the images in 'path'
        if idx is None:
            idx = np.arange(self.num_views)
        elif isinstance(idx, int):
            idx = np.array([idx])
        elif isinstance(idx, torch.Tensor):
            idx = np.asarray(idx.cpu())
        elif isinstance(idx, slice):
            idx = np.arange(self.num_views)[idx]
        if len(idx.shape) < 1:
            idx = np.array([idx])

        # Size to which the images should be reshaped
        if size is None:
            size = self.img_size

        # Rollings of the images
        if rollings is not None:
            assert rollings.dtype == torch.int64, \
                f"Expected dtype=torch.int64 but got dtype={rollings.dtype} " \
                f"instead."
            assert rollings.shape[0] == idx.shape[0], \
                f"Expected tensor of shape {idx.shape[0]} but got " \
                f"{rollings.shape[0]} instead."
        else:
            rollings = torch.zeros(idx.shape[0]).long()

        # Cropping boxes size and offsets
        # XAND(crop_size and crop_offsets)
        assert bool(crop_size) == bool(crop_offsets is not None), \
            f"If either 'crop_size' or 'crop_offsets' is specified, both " \
            f"must be specified."
        if crop_size is not None:
            crop_size = tuple(crop_size)
            assert len(crop_size) == 2, \
                f"Expected len(crop_size)=2 but got {len(crop_size)} instead."
            assert all(a <= b for a, b in zip(crop_size, size)), \
                f"Expected crop_size to be smaller than size but got " \
                f"size={size} and crop_size={crop_size} instead."
            assert crop_offsets.dtype == torch.int64, \
                f"Expected dtype=torch.int64 but got dtype=" \
                f"{crop_offsets.dtype} instead."
            assert crop_offsets.shape == (idx.shape[0], 2), \
                f"Expected tensor of shape {(idx.shape[0], 2)} but got " \
                f"{crop_offsets.shape} instead."
        else:
            crop_size = size
            crop_offsets = torch.zeros((idx.shape[0], 2)).long()

        # Downsampling after cropping
        assert downscale is None or downscale >= 1, \
            f"Expected scalar larger than 1 but got {downscale} instead."

        # Read images from files
        path_enum = tq(self.path[idx]) if show_progress else self.path[idx]
        images = [Image.open(p).convert('RGB').resize(size) for p in path_enum]

        # Local helper to roll a PIL image sideways
        # source: https://pillow.readthedocs.io
        def pil_roll(image, delta):
            xsize, ysize = image.size

            delta = delta % xsize
            if delta == 0:
                return image

            part1 = image.crop((0, 0, delta, ysize))
            part2 = image.crop((delta, 0, xsize, ysize))
            part1.load()
            part2.load()
            image.paste(part2, (0, 0, xsize - delta, ysize))
            image.paste(part1, (xsize - delta, 0, xsize, ysize))
            return image

        # Roll the images
        images = [pil_roll(im, r.item())
                  for im, r in zip(images, rollings.cpu())]

        # Crop and resize
        if downscale is None:
            w, h = crop_size
            images = [im.crop((left, top, left + w, top + h))
                      for im, (left, top)
                      in zip(images, np.asarray(crop_offsets.cpu()))]
        else:
            end_size = tuple(int(x / downscale) for x in crop_size)
            w, h = crop_size
            images = [im.resize(end_size, box=(left, top, left + w, top + h))
                      for im, (left, top)
                      in zip(images, np.asarray(crop_offsets.cpu()))]

        # Convert to torch batch
        images = torch.from_numpy(np.stack([np.asarray(im) for im in images]))
        images = images.permute(0, 3, 1, 2)

        return images

    def __len__(self):
        """Returns the number of image views in the
        SameSettingImageData.
        """
        return self.num_views

    def __getitem__(self, idx):
        """Indexing mechanism.

        Returns a new copy of the indexed SameSettingImageData.
        Supports torch and numpy indexing. For practical reasons, we
        don't want to have duplicate images in the SameSettingImageData,
        so indexing with duplicates
        will raise an error.
        """
        idx = tensor_idx(idx).to(self.device)
        assert idx.unique().numel() == idx.shape[0], \
            f"Index must not contain duplicates."
        idx_numpy = np.asarray(idx.cpu())

        return self.__class__(
            path=self.path[idx_numpy],
            pos=self.pos[idx],
            extrinsic=self.extrinsic[idx],
            cam_size=copy.deepcopy(self.cam_size),
            downscale=copy.deepcopy(self.downscale),
            crop_size=copy.deepcopy(self.crop_size),
            crop_offsets=self.crop_offsets[idx],
            x=self.x[idx] if self.x is not None else None,
            mappings=self.mappings.select_images(idx) if self.mappings is not None else None,
            camera_idx=self.camera_idx[idx] if self.camera_idx else None,
            cameras=self.cameras)

    def __iter__(self):
        """Iteration mechanism.

        Looping over the SameSettingImageData will return an
        SameSettingImageData for each individual image view.
        """
        i: int
        for i in range(self.__len__()):
            yield self[i]

    def __repr__(self):
        return f"{self.__class__.__name__}(num_views={self.num_views}, " \
               f"num_points={self.num_points}, device={self.device})"

    def clone(self):
        """Returns a shallow copy of self, except for 'x' and
        'mappings', which are cloned as they may carry gradients.
        """
        out = copy.copy(self)
        out._x = self.x.clone() if self.x is not None \
            else None
        out._mappings = self.mappings.clone() if self.mappings is not None \
            else None
        return out

    def to(self, device):
        """Set torch.Tensor attributes device."""
        out = self.__class__(
            path=self.path, pos=self.pos.to(device),
            extrinsic=self.extrinsic.to(device), cam_size=self.cam_size,
            downscale=self.downscale, rollings=self.rollings.to(device),
            crop_size=self.crop_size, crop_offsets=self.crop_offsets.to(device),
            camera_idx=self.camera_idx.to(device) if self.camera_idx else None,
            cameras=self.cameras)
        out._x = self.x.to(device) if self.x is not None else None
        out._mappings = self.mappings.to(device) if self.mappings is not None \
            else None
        return out

    @property
    def device(self):
        """Get the device of the torch.Tensor attributes."""
        return self.pos.device

    @property
    def settings_hash(self):
        """Produces a hash of the shared SameSettingImageData settings.
        This hash can be used as an identifier to characterize the
        SameSettingImageData for Batching mechanisms.
        """
        # Assert shared keys are the same for all items
        keys = tuple(set(SameSettingImageData._shared_keys))
        return hash(tuple(getattr(self, k) for k in keys))

    @staticmethod
    def get_batch_type():
        """Required by MMData.from_mm_data_list."""
        return SameSettingImageBatch

    @property
    def feature_map_indexing(self):
        """Return the indices for extracting mapped data from the
        corresponding batch of image feature maps.

        The batch of image feature maps X is expected to have the shape
        `[B, C, H, W]`. The returned indexing object idx is intended to
        be used for recovering the mapped features as: `X[idx]`.
        """
        if self.mappings is not None:
            return self.mappings.feature_map_indexing
        return None

    @property
    def atomic_csr_indexing(self):
        """Return the indices that will be used for atomic-level pooling
        on CSR-formatted data.
        """
        if self.mappings is not None:
            return self.mappings.atomic_csr_indexing
        return None

    @property
    def view_csr_indexing(self):
        """Return the indices that will be used for view-level pooling
        on CSR-formatted data.
        """
        if self.mappings is not None:
            return self.mappings.view_csr_indexing
        return None

    @property
    def mapping_features(self):
        """Return the mapping features carried by the mappings."""
        return self.mappings.features

    def get_mapped_features(self, interpolate=False):
        """Return the mapped features, with optional interpolation. If
        `interpolate=False`, the mappings will be adjusted to
        `self.img_size`: the current size of the feature map `self.x`.
        """
        # Compute the feature map's sampling ratio between the input
        # `mapping_size` and the current `img_size`
        # TODO: treat scales independently. Careful with min or max
        #  depending on upscale and downscale
        scale = 1 / self.downscale

        # If not interpolating, set the mapping to the proper scale
        mappings = self.mappings if interpolate \
            else self.mappings.rescale_images(scale)

        # Index the features with/without interpolation
        if interpolate and scale != 1:
            resolution = torch.Tensor([self.mapping_size]).to(self.device)
            coords = mappings.pixels / (resolution - 1)
            coords = coords[:, [1, 0]]  # pixel mappings are in (W, H) format
            batch = mappings.feature_map_indexing[0]
            x = sparse_interpolation(self.x, coords, batch)
        else:
            x = self.x[mappings.feature_map_indexing]

        return x


class SameSettingImageBatch(SameSettingImageData):
    """Wrapper class of SameSettingImageData to build a batch from a
    list of SameSettingImageData and reconstruct it afterwards.

    Each SameSettingImageData in the batch is assumed to refer to
    different Data objects in its mappings. For that reason, if the
    SameSettingImageData have mappings, they will also be batched with
    their point ids reindexed. For consistency, this implies that
    associated Data points are expected to be batched in the same order.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__sizes__ = None

    @property
    def batch_pointers(self):
        return np.cumsum(np.concatenate(([0], self.__sizes__))) \
            if self.__sizes__ is not None else None

    @property
    def batch_items_sizes(self):
        return self.__sizes__ if self.__sizes__ is not None else None

    @property
    def num_batch_items(self):
        return len(self.__sizes__) if self.__sizes__ is not None else None

    @staticmethod
    def from_data_list(image_data_list):
        assert isinstance(image_data_list, list) and len(image_data_list) > 0
        assert all(isinstance(x, SameSettingImageData)
                   for x in image_data_list)

        # Recover the attributes of the first SameSettingImageData to
        # compare the shared attributes with the other
        # SameSettingImageData
        batch_dict = image_data_list[0].to_dict()
        sizes = [image_data_list[0].num_views]
        for key in SameSettingImageData._own_keys:
            batch_dict[key] = [batch_dict[key]]

        # Only stack if all SameSettingImageData have the same shared
        # attributes
        if len(image_data_list) > 1:

            # Make sure shared keys are the same across the batch
            hash_ref = image_data_list[0].settings_hash
            assert all(im.settings_hash == hash_ref
                       for im in image_data_list), \
                f"All SameSettingImageData values for shared keys " \
                f"{SameSettingImageData._shared_keys} must be the same."

            for image_data in image_data_list[1:]:

                # Prepare stack keys for concatenation or batching
                image_dict = image_data.to_dict()
                for key, value in [(k, v) for (k, v) in image_dict.items()
                                   if k in SameSettingImageData._own_keys]:
                    batch_dict[key] += [value]

                # Prepare the sizes for items recovery with
                # .to_data_list
                sizes.append(image_data.num_views)

        # Concatenate numpy array attributes. Try/except here is needed
        # to avoid crashing in case an attribute is None or non-existent
        for key in SameSettingImageData._numpy_keys:
            try:
                batch_dict[key] = np.concatenate(batch_dict[key])
            except:
                batch_dict[key] = None

        # Concatenate torch tensor attributes. Try/except here is needed
        # to avoid crashing in case an attribute is None or non-existent
        for key in SameSettingImageData._torch_keys + [SameSettingImageData._x_key]:
            try:
                batch_dict[key] = torch.cat(batch_dict[key])
            except:
                batch_dict[key] = None

        # Batch mappings, unless one of the items does not have mappings
        if any(mappings is None
               for mappings in batch_dict[SameSettingImageData._map_key]):
            batch_dict[SameSettingImageData._map_key] = None
        else:
            batch_dict[SameSettingImageData._map_key] = \
                ImageMappingBatch.from_csr_list(
                    batch_dict[SameSettingImageData._map_key])

        # Initialize the batch from dict and keep track of the item
        # sizes
        batch = SameSettingImageBatch(**batch_dict)
        batch.__sizes__ = np.array(sizes)

        return batch

    def to_data_list(self):
        if self.__sizes__ is None:
            raise RuntimeError(
                'Cannot reconstruct image data list from batch because '
                'the batch object was not created using '
                '`SameSettingImageBatch.from_data_list()`.')

        batch_pointers = self.batch_pointers
        return [self[batch_pointers[i]:batch_pointers[i + 1]]
                for i in range(self.num_batch_items)]


class ImageData:
    """Holder for SameSettingImageData items. Useful when
    SameSettingImageData can't be batched together because their
    internal settings differ. Default format for handling image
    attributes, features and mappings in multimodal models and modules.
    """

    def __init__(self, image_list: List[SameSettingImageData]):
        self._list = image_list
        # self.debug()

    @property
    def num_settings(self):
        return len(self)

    @property
    def num_views(self):
        return sum([im.num_views for im in self])

    @property
    def num_points(self):
        return self[0].num_points if len(self) > 0 else 0

    @property
    def x(self):
        return [im.x for im in self]

    @x.setter
    def x(self, x_list):
        assert x_list is None or isinstance(x_list, list), \
            f"Expected a List but got {type(x_list)} instead."

        if x_list is None or len(x_list) == 0:
            x_list = [None] * self.num_settings

        for im, x in zip(self, x_list):
            im.x = x

    def debug(self):
        assert isinstance(self._list, list), \
            f"Expected a list of SameSettingImageData but got " \
            f"{type(self._list)} instead."
        assert all(isinstance(im, SameSettingImageData) for im in self), \
            f"All list elements must be of type SameSettingImageData."
        # Remove any empty SameSettingImageData from the list
        #         self._list = [im for im in self._list if im.num_views > 0]
        assert all(im.num_points == self.num_points for im in self), \
            "All SameSettingImageData mappings must refer to the same Data. " \
            "Hence, all must have the same number of points in their mappings."
        assert len(set([im.settings_hash for im in self])) == len(self), \
            "All SameSettingImageData in ImageData must have " \
            "different settings. SameSettingImageData with the same " \
            "settings are expected to be grouped together in the same " \
            "SameSettingImageData.)"
        for im in self:
            im.debug()

    def __len__(self):
        return len(self._list)

    def __getitem__(self, idx):
        if self.__len__() == 0:
            raise ValueError(
                f'{self} cannot be indexed because it has length 0.')

        # TODO: only return self.__class__ data from this ? Isn't it
        #  awkward to change classes for integer indexation only ?
        if isinstance(idx, int) and idx < self.__len__():
            return self._list[idx]
        else:
            return self.__class__([
                self._list[i] for i in tensor_idx(idx).tolist()])

    def __iter__(self):
        for i in range(self.__len__()):
            yield self[i]

    def __repr__(self):
        return f"{self.__class__.__name__}(num_settings={self.num_settings}, " \
               f"num_views={self.num_views}, num_points={self.num_points}, " \
               f"device={self.device})"

    def select_points(self, idx, mode='pick'):
        return self.__class__([
            im.select_points(idx, mode=mode) for im in self])

    def select_views(self, view_mask_list):
        assert isinstance(view_mask_list, list), \
            "Expected a list of view masks."
        return self.__class__([
            im.select_views(view_mask)
            for im, view_mask in zip(self, view_mask_list)])

    def load(self):
        self._list = [im.load() for im in self]
        return self

    def clone(self):
        return self.__class__([im.clone() for im in self])

    def to(self, device):
        out = self.clone()
        out._list = [im.to(device) for im in out]
        return out

    @property
    def device(self):
        return self[0].device if len(self) > 0 else 'cpu'

    @staticmethod
    def get_batch_type():
        """Required by MMData.from_mm_data_list."""
        return ImageBatch

    def get_mapped_features(self, interpolate=False):
        """Return the list of mapped features for each image, with
        optional interpolation. If `interpolate=False`, the mappings
        will be adjusted to `self.img_size`: the current size of the
        feature map `self.x`.
        """
        return [im.get_mapped_features(interpolate=interpolate) for im in self]

    @property
    def feature_map_indexing(self):
        """Return the indices for extracting mapped data from the
        corresponding batch of image feature maps.

        The batch of image feature maps X is expected to have the shape
        `[B, C, H, W]`. The returned indexing object idx is intended to
        be used for recovering the mapped features as: `X[idx]`.
        """
        return [im.feature_map_indexing for im in self]

    @property
    def atomic_csr_indexing(self):
        """Return the indices that will be used for atomic-level pooling
        on CSR-formatted data.
        """
        return [im.atomic_csr_indexing for im in self]

    @property
    def view_cat_sorting(self):
        """Return the sorting indices to arrange concatenated view-level
        features to a CSR-friendly order wrt to points.
        """
        # Recover the expanded view idx for each SameSettingImageData
        # in self
        dense_idx_list = [
            torch.arange(im.num_points, device=self.device).repeat_interleave(
                im.view_csr_indexing[1:] - im.view_csr_indexing[:-1])
            for im in self]

        try:
            # Assuming the corresponding view features will be concatenated
            # in the same order as in self, compute the sorting indices to
            # arrange features wrt point indices, to facilitate CSR indexing
            sorting = torch.cat([idx for idx in dense_idx_list]).argsort()
        except:
            print(f'self : {self}')
            print(f'len(self) : {len(self)}')
            print(f'dense_idx_list : {dense_idx_list}')
            print(f'num_points : {[im.num_points for im in self]}')
            print(f'view_csr_indexing : {[im.view_csr_indexing for im in self]}')
            raise ValueError

        return sorting

    @property
    def view_cat_csr_indexing(self):
        """Return the indices that will be used for view-level pooling
        on CSR-formatted data. To sort concatenated view-level features,
        see 'view_cat_sorting'.
        """
        # Assuming the features have been concatenated and sorted as
        # aforementioned in 'view_cat_sorting' compute the new CSR
        # indices to be used for feature view-pooling
        view_csr_idx = torch.cat([
            im.view_csr_indexing.unsqueeze(dim=1)
            for im in self], dim=1).sum(dim=1)
        return view_csr_idx

    @property
    def mapping_features(self):
        """Return the mapping features carried by the mappings of each
        SameSettingImageData.
        """
        return [im.mapping_features for im in self]


class ImageBatch(ImageData):
    """Wrapper class of ImageData to build a batch from a list
    of ImageData and reconstruct it afterwards.

    Like SameSettingImageBatch, each ImageData in the batch here is
    assumed to refer to different Data objects. Hence, the point ids in
    ImageBatch mappings are reindexed. For consistency, this
    implies that associated Data points are expected to be batched in
    the same sorder.
    """

    def __init__(self, image_list: List[SameSettingImageData]):
        super().__init__(image_list)
        self.__il_sizes__ = None
        self.__hashes__ = None
        self.__il_idx_dict__ = None
        self.__im_idx_dict__ = None
        self.__cum_pts__ = None

    @staticmethod
    def from_data_list(image_data_list):
        assert isinstance(image_data_list, list) and len(image_data_list) > 0
        assert all(isinstance(x, ImageData) for x in image_data_list)

        # Recover the list of unique hashes
        hashes = list(set([
            im.settings_hash
            for il in image_data_list
            for im in il]))
        hashes_idx = {h: i for i, h in enumerate(hashes)}

        # Recover the number of points in each ImageData
        n_pts = torch.LongTensor([il.num_points for il in image_data_list])
        cum_pts = torch.cumsum(torch.cat(
            (torch.LongTensor([0]), n_pts)), dim=0)

        # Recover the size of each input ImageData
        il_sizes = [len(il) for il in image_data_list]

        # ImageData idx in input list
        il_idx_dict = {h: [] for h in hashes}

        # SameSettingImageData idx in ImageData
        im_idx_dict = {h: [] for h in hashes}

        # Distribute the SameSettingImageData to its relevant hash
        batches = [[]] * len(hashes)
        for il_idx, il in enumerate(image_data_list):
            for im_idx, im in enumerate(il):
                h = im.settings_hash
                il_idx_dict[h].append(il_idx)
                im_idx_dict[h].append(im_idx)
                batches[hashes_idx[h]] = batches[hashes_idx[h]] + [im]

        # Batch the SameSettingImageData for each hash
        batches = [SameSettingImageBatch.from_data_list(x) for x in batches]

        # Update the ImageBatches' mappings pointers to account for
        # global points reindexing
        for h, im in zip(hashes, batches):
            if im.num_points > 0:
                global_idx = torch.cat(
                    [torch.arange(cum_pts[il_idx], cum_pts[il_idx + 1])
                     for il_idx in il_idx_dict[h]], dim=0)
                im.mappings.insert_empty_groups(global_idx,
                                                num_groups=cum_pts[-1])

        msi_batch = ImageBatch(batches)
        msi_batch.__il_sizes__ = il_sizes
        msi_batch.__hashes__ = hashes
        msi_batch.__il_idx_dict__ = il_idx_dict
        msi_batch.__im_idx_dict__ = im_idx_dict
        msi_batch.__cum_pts__ = cum_pts

        return msi_batch

    def to_data_list(self):
        assert (self.__il_sizes__ is not None
                and self.__hashes__ is not None
                and self.__il_idx_dict__ is not None
                and self.__im_idx_dict__ is not None
                and self.__cum_pts__ is not None), \
            "Cannot reconstruct the list of MultiSettingImages because " \
            "the ImageBatch was not created using " \
            "'ImageBatch.from_data_list'."

        # Initialize the MultiSettingImages
        msi_list = [[None] * s for s in self.__il_sizes__]

        for h, ib in zip(self.__hashes__, self):
            # Restore the individual SameSettingImageData from the
            # SameSettingImageBatch
            for il_idx, im_idx, im in zip(
                    self.__il_idx_dict__[h],
                    self.__im_idx_dict__[h],
                    ib.to_data_list()):
                # Restore the point ids in the mappings
                start = self.__cum_pts__[il_idx]
                end = self.__cum_pts__[il_idx + 1]
                im.mappings = im.mappings[start:end]

                # Update the list of MultiSettingImages with each
                # SameSettingImageData in its original position
                msi_list[il_idx][im_idx] = im

        # Convert to MultiSettingImage
        return [ImageData(x) for x in msi_list]


class ImageMapping(CSRData):
    """CSRData format for point-image-pixel mappings.

    Example
    -------
    import torch
    from torch_points3d.core.multimodal.image import ImageMapping
    from torch_points3d.core.multimodal.csr import CSRData

    n_points = 3
    n_views = 12
    n_pixels = 1000

    indices = torch.sort(torch.randint(0, size=(n_views,), high=n_points))[0]
    float_values = torch.rand(n_views)
    indices_nested = torch.sort(torch.randint(0, size=(n_pixels,), high=n_views))[0]
    csr_nested = CSRData(indices_nested, torch.arange(indices_nested.shape[0]), dense=True)

    ImageMapping(indices, float_values, csr_nested, dense=True)
    """

    @staticmethod
    def from_dense(point_ids, image_ids, pixels, features, num_points=None):
        """Recommended method for building an ImageMapping from dense
        data.
        """
        assert point_ids.ndim == 1, \
            'point_ids and image_ids must be 1D tensors'
        assert point_ids.shape == image_ids.shape, \
            'point_ids and image_ids must have the same shape'
        assert point_ids.shape[0] == pixels.shape[0], \
            'pixels and indices must have the same shape'
        assert features is None or point_ids.shape[0] == features.shape[0], \
            'point_ids and features must have the same shape'

        # Sort by point_ids first, image_ids second
        idx_sort = lexargsort(point_ids, image_ids)
        image_ids = image_ids[idx_sort]
        point_ids = point_ids[idx_sort]
        pixels = pixels[idx_sort]
        if features is not None:
            features = features[idx_sort]
        del idx_sort

        # Convert to "nested CSRData" format.
        # Compute point-image pointers in the pixels array.
        # NB: The pointers are marked by non-successive point-image ids.
        #     Watch out for overflow in case the point_ids and
        #     image_ids are too large and stored in 32 bits.
        composite_ids = CompositeTensor(
            point_ids, image_ids, device=point_ids.device)
        image_pixel_mappings = CSRData(composite_ids.data, pixels, dense=True)
        del composite_ids

        # Compress point_ids and image_ids by taking the last value of
        # each pointer. For features, take the mean across the pixel
        # masks
        image_ids = image_ids[image_pixel_mappings.pointers[1:] - 1]
        point_ids = point_ids[image_pixel_mappings.pointers[1:] - 1]
        if features is not None:
            features = torch_scatter.segment_csr(
                features, image_pixel_mappings.pointers, reduce='mean')

        # Instantiate the main CSRData object
        # Compute point pointers in the image_ids array
        if features is None:
            mapping = ImageMapping(
                point_ids, image_ids, image_pixel_mappings, dense=True,
                is_index_value=[True, False])
        else:
            mapping = ImageMapping(
                point_ids, image_ids, image_pixel_mappings, features,
                dense=True, is_index_value=[True, False, False])

        # Some points may have been seen by no image so we need to
        # inject 0-sized pointers to account for these.
        # NB: we assume all relevant points are present in
        # range(num_points), if a point with an id larger than
        # num_points were to exist, we would not be able to take it
        # into account in the pointers.
        if num_points is None or num_points < point_ids.max() + 1:
            num_points = point_ids.max() + 1

        # Compress point_ids by taking the last value of each pointer
        point_ids = point_ids[mapping.pointers[1:] - 1]
        mapping = mapping.insert_empty_groups(
            point_ids, num_groups=num_points)

        return mapping

    def debug(self):
        # CSRData debug
        super().debug()

        # ImageMapping-specific debug
        assert len(self.values) == 2 or self.has_features, \
            f"CSRData format does not match that of ImageMapping: " \
            f"len(values) should be 2 or 3 but is {len(self.values)}."
        assert isinstance(self.values[1], CSRData), \
            f"CSRData format does not match that of ImageMapping: " \
            f"values[1] is {type(self.values[1])} but should inherit " \
            f"from CSRData"
        assert len(self.values[1].values) == 1, \
            f"CSRData format does not match that of ImageMapping: " \
            f"len(values[1].values) should be 1 but is " \
            f"{len(self.values[1].values)}."

    @property
    def points(self):
        return torch.arange(self.num_groups, device=self.device)

    @property
    def images(self):
        return self.values[0]

    @images.setter
    def images(self, images):
        self.values[0] = images.to(self.device)

    @property
    def has_features(self):
        return len(self.values) == 3

    @property
    def features(self):
        return self.values[2] if self.has_features else None

    @features.setter
    def features(self, features):
        if self.has_features:
            if features is None:
                self.values.pop(-1)
            else:
                self.values[2] = features.to(self.device)
        else:
            if features is not None:
                self.values.append(features.to(self.device))
                # self.debug()

    @property
    def pixels(self):
        return self.values[1].values[0]

    @pixels.setter
    def pixels(self, pixels):
        self.values[1].values[0] = pixels.to(self.device)

    @staticmethod
    def get_batch_type():
        """Required by CSRBatch.from_csr_list."""
        return ImageMappingBatch

    @property
    def bounding_boxes(self):
        """Return the (w_min, w_max, h_min, h_max) pixel values per
        image.
        """
        # TODO: handle circular panoramic images and relevant cropping
        image_ids = self.images.repeat_interleave(
            self.values[1].pointers[1:] - self.values[1].pointers[:-1])
        min_pix, _ = torch_scatter.scatter_min(self.pixels, image_ids, dim=0)
        max_pix, _ = torch_scatter.scatter_max(self.pixels, image_ids, dim=0)
        return min_pix[:, 0], max_pix[:, 0], min_pix[:, 1], max_pix[:, 1]

    @property
    def feature_map_indexing(self):
        """Return the indices for extracting mapped data from the
        corresponding batch of image feature maps.

        The batch of image feature maps X is expected to have the shape
        `[B, C, H, W]`. The returned indexing object idx is intended to
        be used for recovering the mapped features as: `X[idx]`.
        """
        idx_batch = self.images.repeat_interleave(
            self.values[1].pointers[1:] - self.values[1].pointers[:-1])
        idx_height = self.pixels[:, 1]
        idx_width = self.pixels[:, 0]
        idx = (idx_batch.long(), ..., idx_height.long(), idx_width.long())
        return idx

    @property
    def atomic_csr_indexing(self):
        """Return the indices that will be used for atomic-level pooling
        on CSR-formatted data.
        """
        return self.values[1].pointers

    @property
    def view_csr_indexing(self):
        """Return the indices that will be used for view-level pooling
        on CSR-formatted data.
        """
        return self.pointers

    def rescale_images(self, ratio):
        """Update the image resolution after resampling. Typically
        called after a downsampling or upsampling layer in an image CNN
        module.

        The mappings will be downsampled if `ratio < 1`, and they will
        be upsampled if `ratio > 1`.

        Returns a new ImageMapping object.
        """
        if ratio < 1:
            return self.downscale_images(1 / ratio)
        else:
            return self.upscale_images(ratio)

    def downscale_images(self, ratio):
        """Update the image resolution after subsampling. Typically
        called after a pooling layer in an image CNN encoder.

        To update the image resolution in the mappings, the pixel
        coordinates are converted to lower resolutions. This operation
        is likely to produce duplicates. Searching and removing these
        duplicates only affects the atomic-level mappings, so only the
        pixel-level nested CSRData is modified by this function.

        Returns a new ImageMapping object.
        """
        assert ratio >= 1, \
            f"Invalid image subsampling ratio: {ratio}. Must be larger than 1."

        # Create a copy of self
        out = self.clone()

        # Save time when the sampling did not change
        if ratio == 1:
            return out

        # Expand atomic-level mappings to 'dense' format
        ids = torch.arange(
            out.values[1].num_groups, device=self.device).repeat_interleave(
            out.values[1].pointers[1:] - out.values[1].pointers[:-1])
        pix_x = out.values[1].values[0][:, 0]
        pix_y = out.values[1].values[0][:, 1]
        pix_dtype = pix_x.dtype

        # Convert pixel coordinates to new resolution
        pix_x = (pix_x // ratio).long()
        pix_y = (pix_y // ratio).long()

        # Remove duplicates and sort wrt ids
        # Assuming this does not cause issues for other potential
        # atomic-level CSR-nested values
        idx_unique = lexargunique(ids, pix_x, pix_y)
        ids = ids[idx_unique]
        pix_x = pix_x[idx_unique]
        pix_y = pix_y[idx_unique]

        # Build the new atomic-level CSR mapping
        if isinstance(out.values[1], CSRBatch):
            sizes = out.values[1].__sizes__
            out.values[1] = CSRBatch(
                ids, torch.stack((pix_x, pix_y), dim=1).type(pix_dtype),
                dense=True)
            out.values[1].__sizes__ = sizes
        elif isinstance(out.values[1], CSRData):
            out.values[1] = CSRData(
                ids, torch.stack((pix_x, pix_y), dim=1).type(pix_dtype),
                dense=True)
        else:
            raise NotImplementedError(
                "The atomic-level mappings must be either a CSRData or "
                "CSRBatch object.")

        return out

    def upscale_images(self, ratio, center=True):
        """Update the image resolution after upsampling. Typically
        called after an upsampling layer in an image CNN decoder.

        To update the image resolution in the mappings, the pixel
        coordinates are converted to higher resolutions. If
        `center=True`, the higher-resolution pixel in chosen to be the
        closest to the center of the lower-resolution pixel. If
        `center=False`, the higher-resolution coordinates correspond to
        the top-left corner of the lower-resolution pixel.

        Note that this operation is not strictly the inverse of
        `self.downscale_image`. Indeed, the latter discards redundant
        mappings and loses spatial precision that cannot be recovered
        in `self.upscale_images`.

        Returns a new ImageMapping object.
        """
        assert ratio >= 1, \
            f"Invalid image upsampling ratio: {ratio}. Must be larger than 1."

        # Create a copy of self
        out = self.clone()

        # Save time when the sampling did not change
        if ratio == 1:
            return out

        # Recover pixel coordinates
        pix_x = out.values[1].values[0][:, 0]
        pix_y = out.values[1].values[0][:, 1]
        pix_dtype = pix_x.dtype

        # Convert pixel coordinates to new resolution
        if center:
            pix_x = (pix_x.float() * ratio + ratio / 2).long()
            pix_y = (pix_y.float() * ratio + ratio / 2).long()
        else:
            pix_x = (pix_x.float() * ratio).long()
            pix_y = (pix_y.float() * ratio).long()

        # Save pixel coordinates in output mapping
        pixels = torch.stack((pix_x, pix_y), dim=1).type(pix_dtype)
        out.values[1].values[0] = pixels

        return out

    def select_images(self, idx):
        """Return a copy of self with images selected with idx.

        Idx is assumed to refer to image indices. The mappings are
        updated so as to remove mappings to image indices absent from
        idx and change the image indexing to respect the new order
        implied by idx: idx[i] -> i.

        For the mappings to preserve their meaning, this operation
        assumes the same indexation is also applied to the
        corresponding SameSettingImageData and contains no duplicate
        indices.
        """
        idx = tensor_idx(idx).to(self.device)
        assert idx.unique().numel() == idx.shape[0], \
            f"Index must not contain duplicates."

        # Rule out empty mappings
        if self.num_items == 0:
            return self.clone()

        # Get view-level indices for images to keep
        view_idx = torch.where((self.images[..., None] == idx).any(-1))[0]

        # Index the values
        values = [val[view_idx] for val in self.values]

        # If idx is empty, return an empty mapping
        if idx.shape[0] == 0:
            out = self.__class__(
                torch.zeros_like(self.pointers), *values, dense=False,
                is_index_value=self.is_index_value)
            # out.debug()
            return out

        # Update the image indices. To do so, create a tensor of indices
        # idx_gen so that the desired output can be computed with simple
        # indexation idx_gen[images]. This avoids using map() or
        # numpy.vectorize alternatives.
        idx_gen = torch.full(
            (idx.max() + 1,), -1, dtype=torch.int64, device=self.device)
        idx_gen = idx_gen.scatter_(
            0, idx, torch.arange(idx.shape[0], device=self.device))
        values[0] = idx_gen[values[0]]  # values[0] holds image indices

        # Update the pointers
        point_ids = torch.arange(
            self.num_groups, device=self.device).repeat_interleave(
            self.pointers[1:] - self.pointers[:-1])
        point_ids = point_ids[view_idx]
        pointers = CSRData._sorted_indices_to_pointers(point_ids)

        # Create the output mapping object
        out = self.__class__(
            pointers, *values, dense=False, is_index_value=self.is_index_value)

        # Some points may have been seen by no image so we need to
        # inject 0-sized pointers to account for these. To get the real
        # point_ids take the last value of each pointer.
        point_ids = point_ids[out.pointers[1:] - 1]
        out = out.insert_empty_groups(point_ids, num_groups=self.num_groups)

        # out.debug()

        return out

    def select_views(self, view_mask):
        """Return a copy of self with views selected with view_mask, as
        well as the corresponding selected image indices.

        So as to preserve the views ordering, view_mask is assumed to be
        a boolean mask over views.

        The mappings are updated so as to remove views to images absent
        from the view_mask and change the image indexing to respect the
        new order.

        For the mappings to preserve their meaning, this operation
        assumes the same indexation is also applied to the corresponding
        SameSettingImageData and contains no duplicate indices.
        """
        if isinstance(view_mask, np.ndarray):
            view_mask = torch.from_numpy(view_mask)
        assert isinstance(view_mask, torch.BoolTensor) \
               and view_mask.dim() == 1 \
               and view_mask.shape[0] == self.num_items, \
            f"view_mask must be a torch.BoolTensor of size {self.num_items}."

        # Rule out empty mappings
        if self.num_items == 0:
            return self.clone()

        # Index the values
        values = [val[view_mask] for val in self.values]

        # If view_mask is empty, return an empty mapping
        if not torch.any(view_mask):
            out = self.__class__(
                torch.zeros_like(self.pointers), *values, dense=False,
                is_index_value=self.is_index_value)
            # out.debug()
            return out, torch.LongTensor([])

        # If need be, update the image indices. To do so, create a
        # tensor of indices idx_gen so that the desired output can be
        # computed with simple indexation idx_gen[images]. This avoids
        # using map() or numpy.vectorize alternatives.
        img_idx = values[0].unique()  # values[0] holds image indices
        if img_idx.numel() < self.images.max() + 1:
            idx_gen = torch.full(
                (img_idx.max() + 1,), -1, dtype=torch.int64, device=self.device)
            idx_gen = idx_gen.scatter_(
                0, img_idx, torch.arange(img_idx.shape[0], device=self.device))
            values[0] = idx_gen[values[0]]
        else:
            img_idx = None

        # Update the pointers
        point_ids = torch.arange(
            self.num_groups, device=self.device).repeat_interleave(
            self.pointers[1:] - self.pointers[:-1])
        point_ids = point_ids[view_mask]
        pointers = CSRData._sorted_indices_to_pointers(point_ids)

        # Create the output mapping object
        out = self.__class__(
            pointers, *values, dense=False, is_index_value=self.is_index_value)

        # Some points may have been seen by no image so we need to
        # inject 0-sized pointers to account for these. To get the real
        # point_ids take the last value of each pointer.
        point_ids = point_ids[out.pointers[1:] - 1]
        out = out.insert_empty_groups(point_ids, num_groups=self.num_groups)

        # out.debug()

        return out, img_idx

    def select_points(self, idx, mode='pick'):
        """Update the 3D points sampling. Typically called after a 3D
        sampling or strided convolution layer in a 3D CNN encoder.

        To update the 3D resolution in the modality data and mappings,
        two methods - ruled by the `mode` parameter - may be used:
        picking or merging.

          - 'pick' (default): only a subset of points is sampled, the
            rest is discarded. In this case, a 1D indexing array must be
            provided.

          - 'merge': points are agglomerated. The mappings are combined,
            duplicates are removed. If any other value (such as
            mapping features) is present in the mapping, the value
            of one of the duplicates is picked at random. In this case,
            the correspondence map for the N points in the mapping must
            be provided as a 1D array of size N.

        Returns a new ImageMapping object.
        """
        # TODO: make sure the merge mode works on real data...
        MODES = ['pick', 'merge']
        assert mode in MODES, \
            f"Unknown mode '{mode}'. Supported modes are {MODES}."

        # Convert idx to a convenient indexing format
        idx = tensor_idx(idx).to(self.device)

        # Work on a clone of self, to avoid in-place modifications.
        # Images are not affected if no mappings are present or idx is
        # None
        if idx is None or idx.shape[0] == 0 or self.num_groups == 0:
            return self.clone()

        # If mappings have no data, we must still update the pointers
        if self.num_items == 0:
            out = self.clone()
            out.pointers = torch.zeros(idx.shape[0] + 1).long().to(self.device)
            return out

        # Picking mode by default
        if mode == 'pick':
            out = self[idx]

        # Merge mode
        elif mode == 'merge':
            try:
                assert idx.shape[0] == self.num_groups > 0, \
                    f"Merge correspondences has size {idx.shape[0]} but size " \
                    f"{self.num_groups} was expected."
                assert (torch.arange(idx.max() + 1, device=self.device)
                        == torch.unique(idx)).all(), \
                    "Merge correspondences must map to a compact set of " \
                    "indices."
            except:
                # TODO: quick fix because we don't know why this occasionally crashes
                return self.clone()

            # Expand to dense view-level format
            point_ids = idx.repeat_interleave(
                self.pointers[1:] - self.pointers[:-1])
            image_ids = self.images

            # Merge view-level mapping features. Take special care for
            # cases when there is no mappings or only a single mapping
            if not self.has_features:
                features = None
            elif self.num_items <= 1:
                features = self.features
            else:
                # Compute composite point-image views ids
                view_ids = CompositeTensor(
                    point_ids, image_ids, device=point_ids.device)
                view_ids = view_ids.data.squeeze()
                # Average the features per view
                features = torch_scatter.scatter_mean(
                    self.features, view_ids, 0)
                # Prepare view indices for torch.gather
                if features.dim() > 1:
                    view_ids = view_ids.view(-1, 1).repeat(1, features.shape[1])
                # Redistribute mean features to source indices
                features = features.gather(0, view_ids)
                del view_ids

            # Expand to dense atomic-level format
            point_ids = idx.repeat_interleave(
                self.pointers[1:] - self.pointers[:-1])
            point_ids = point_ids.repeat_interleave(
                self.values[1].pointers[1:] - self.values[1].pointers[:-1])
            image_ids = self.images.repeat_interleave(
                self.values[1].pointers[1:] - self.values[1].pointers[:-1])
            if self.has_features:
                features = features.repeat_interleave(
                    self.values[1].pointers[1:] - self.values[1].pointers[:-1],
                    dim=0)
            pixels = self.pixels

            # Remove duplicate pixel mappings and aggregate
            idx_unique = lexargunique(
                point_ids, image_ids, pixels[:, 0], pixels[:, 1])
            point_ids = point_ids[idx_unique]
            image_ids = image_ids[idx_unique]
            pixels = pixels[idx_unique]
            features = features[idx_unique] if features is not None else None

            # Convert to CSR format
            out = ImageMapping.from_dense(
                point_ids, image_ids, pixels, features,
                num_points=idx.max() + 1)
        else:
            raise ValueError(f"Unknown point selection mode '{mode}'.")

        return out

    def crop(self, crop_size, crop_offsets):
        """Return a copy of self with cropped image mappings.

        The mappings are updated so as to change pixel coordinates to
        account for a cropping of the mapped images. Each image has its
        own cropping offset, but all share the same cropping box size.

        Pixels discarded by the cropping will also be discarded from
        the mapping.

        For the mappings to preserve their meaning, this operation
        assumes the same cropping is also applied to the corresponding
        SameSettingImageData.
        """
        assert crop_offsets.shape == (self.images.unique().numel(), 2), \
            f"Expected crop_offsets to have shape " \
            f"{(self.images.unique().numel(), 2)} but got shape " \
            f"{crop_offsets.shape} instead."

        # Distribute the offsets to the pixels
        #   - Crop offsets have format: (W, H)
        #   - Pixels have format: (W, H)
        image_ids = self.images.repeat_interleave(
            self.values[1].pointers[1:] - self.values[1].pointers[:-1])
        offsets = crop_offsets[image_ids]
        pixels = self.pixels - offsets

        # Identify the pixels outside of the crop_size box
        #   - Crop size has format: (W, H)
        #   - Pixels have format: (W, H)
        cropped_in_idx = torch.where(
            torch.ge(pixels, torch.Tensor((0, 0))).all(dim=1)
            & torch.lt(pixels, torch.Tensor(crop_size)).all(dim=1))

        # Return if no pixel mapping was cropped out
        if cropped_in_idx[0].shape[0] == 0:
            out = self.clone()
            out.pixels = pixels
            return out

        # Expand to dense format
        point_ids = torch.arange(
            self.num_groups, device=self.device).repeat_interleave(
            self.pointers[1:] - self.pointers[:-1])
        point_ids = point_ids.repeat_interleave(
            self.values[1].pointers[1:] - self.values[1].pointers[:-1])
        # image_ids = self.images.repeat_interleave(
        #     self.values[1].pointers[1:] - self.values[1].pointers[:-1])
        if self.has_features:
            features = self.features.repeat_interleave(
                self.values[1].pointers[1:] - self.values[1].pointers[:-1],
                dim=0)
        else:
            features = None

        # Select only the valid mappings and create a mapping
        point_ids = point_ids[cropped_in_idx]
        image_ids = image_ids[cropped_in_idx]
        features = features[cropped_in_idx] if features is not None else None
        pixels = pixels[cropped_in_idx]

        # Convert to CSR format
        return ImageMapping.from_dense(
            point_ids, image_ids, pixels, features, num_points=self.num_groups)


class ImageMappingBatch(ImageMapping, CSRBatch):
    """Batch wrapper for ImageMapping."""
    __csr_type__ = ImageMapping


"""

import torch
from torch_points3d.core.multimodal.csr import *
from torch_points3d.core.multimodal.image import *
from torch_points3d.utils.multimodal import lexsort

n_groups = 10**5
n_items = 10**6
idx = torch.randint(low=0, high=n_groups, size=(n_items,))
img_idx = torch.randint(low=0, high=3, size=(n_items,))
pixels = torch.randint(low=0, high=10, size=(n_items,2))
features = torch.rand(n_items, 3)

idx, img_idx = lexsort(idx, img_idx)

m = ImageMapping.from_dense(idx, img_idx, pixels, features)

b = ImageMappingBatch.from_csr_list([m[2], m[1:3], m, m[0]])

a = m[2].num_groups + m[1:3].num_groups
print((b[a : a + m.num_groups].values[1].values[0] == m.values[1].values[0]).all().item())

print((b.to_csr_list()[2].pointers == m.pointers).all().item())
print((b.to_csr_list()[2].values[1].values[0] == m.values[1].values[0]).all().item())

b[[0,0,1]]

b = CSRBatch.from_csr_list([m[2], m[1:3], m, m[0]])

#-----------------------------------------------

pointers = torch.LongTensor([0, 0,  5, 12, 12, 15])
val = torch.arange(15)
m = CSRData(pointers, val, dense=False)
b = CSRBatch.from_csr_list([m, m, m])

# b[[0, 1, 7, 8, 14]]
b[[0,0,5]]

"""
