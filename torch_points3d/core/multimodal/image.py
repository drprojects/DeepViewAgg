import copy
import numpy as np
from PIL import Image
import torch
import torch_scatter
from typing import List
from torch_points3d.core.multimodal import CSRData, CSRBatch
from torch_points3d.utils.multimodal import lexargsort, lexunique, \
    lexargunique, CompositeTensor
from torch_points3d.utils.multimodal import tensor_idx



class SameSettingImageData(object):
    """
    Class to hold arrays of images information, along with shared 3D-2D 
    projection information.

    Attributes
        path:numpy.ndarray      image paths
        pos:torch.Tensor        image positions
        opk:torch.Tensor        image angular poses

        ref_size:tuple          initial size of the loaded images and mappings
        proj_upscale:float      upsampling of projection wrt to ref_size
        downscale:float         downsampling of images and mappings wrt ref_size
        rollings:LongTensor     rolling offsets for each image wrt ref_size
        crop_size:tuple         size of the cropping box wrt ref_size
        crop_offsets:LongTensor cropping box offsets for each image wrt ref_size

        voxel:float             voxel resolution
        r_max:float             radius max
        r_min:float             radius min
        growth_k:float          pixel growth factor
        growth_r:float          pixel growth radius

        x:Tensor                tensor of features
        mappings:ImageMapping   mappings between 3D points and the images
        mask:BoolTensor         projection mask
    """
    _numpy_keys = ['path']
    _torch_keys = ['pos', 'opk', 'crop_offsets', 'rollings']
    _map_key = 'mappings'
    _x_key = 'x'
    _mask_key = 'mask'
    _shared_keys = ['ref_size', 'proj_upscale', 'downscale', 'crop_size',
                    'voxel', 'r_max', 'r_min', 'growth_k', 'growth_r',
                    _mask_key]
    _own_keys = _numpy_keys + _torch_keys + [_map_key, _x_key]
    _keys = _shared_keys + _own_keys

    def __init__(
            self, path=np.empty(0, dtype='O'), pos=torch.empty([0, 3]),
            opk=torch.empty([0, 3]), ref_size=(512, 256), proj_upscale=2,
            downscale=1, rollings=None, crop_size=None, crop_offsets=None,
            voxel=0.1, r_max=30, r_min=0.5, growth_k=0.2, growth_r=10, x=None,
            mappings=None, mask=None, **kwargs):

        assert path.shape[0] == pos.shape[0] == opk.shape[0], \
            f"Attributes 'path', 'pos' and 'opk' must have the same length."

        # Initialize the private internal state attributes
        self._ref_size = None
        self._proj_upscale = None
        self._rollings = None
        self._downscale = None
        self._crop_size = None
        self._crop_offsets = None
        self._x = None
        self._mappings = None
        self._mask = None

        # Initialize from parameters
        self.path = np.array(path)
        self.pos = pos.double()
        self.opk = opk.double()
        self.ref_size = ref_size
        self.proj_upscale = proj_upscale
        self.rollings = rollings if rollings is not None \
            else torch.zeros(self.num_views, dtype=torch.int64)
        self.crop_size = crop_size if crop_size is not None else self.ref_size
        self.crop_offsets = crop_offsets if crop_offsets is not None \
            else torch.zeros((self.num_views, 2), dtype=torch.int64)
        self.downscale = downscale
        self.voxel = voxel
        self.r_max = r_max
        self.r_min = r_min
        self.growth_k = growth_k
        self.growth_r = growth_r
        self.x = x
        self.mappings = mappings
        self.mask = mask

        self.debug()

    def debug(self):
        assert self.path.shape[0] == self.pos.shape[0] == self.opk.shape[0], \
            f"Attributes 'path', 'pos' and 'opk' must have the same length."
        assert self.pos.device == self.opk.device, \
            f"Discrepancy in the devices of 'pos' and 'opk' attributes. " \
            f"Please use `SameSettingImageData.to()` to set the device."
        assert len(tuple(self.ref_size)) == 2, \
            f"Expected len(ref_size)=2 but got {len(self.ref_size)} instead."
        assert self.proj_upscale >= 1, \
            f"Expected scalar larger than 1 but got {self.proj_upscale} " \
            f"instead."
        assert self.rollings.shape[0] == self.num_views, \
            f"Expected tensor of size {self.num_views} but got " \
            f"{self.rollings.shape[0]} instead."
        assert len(tuple(self.crop_size)) == 2, \
            f"Expected len(crop_size)=2 but got {len(self.crop_size)} instead."
        assert all(a <= b for a, b in zip(self.crop_size, self.ref_size)), \
            f"Expected size smaller than {self.ref_size} but got " \
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
            img_idx = torch.arange(self.num_views)
            assert (unique_idx == img_idx).all(), \
                f"Image indices in the mappings do not match the " \
                f"SameSettingImageData image indices."
            w_max, h_max = self.mappings.pixels.max(dim=0).values
            assert w_max < self.img_size[0] and h_max < self.img_size[1], \
                f"Max pixel values should be smaller than ({self.img_size}) " \
                f"but got ({w_max, h_max}) instead."
            assert self.device == self.mappings.device, \
                f"Discrepancy in the devices of 'pos' and 'mappings' " \
                f"attributes. Please use `SameSettingImageData.to()` to set " \
                f"the device."
            self.mappings.debug()

        if self.mask is not None:
            assert self.mask.dtype == torch.bool, \
                f"Expected a dtype=torch.bool but got dtype=" \
                f"{self.mask.dtype} instead."
            assert self.mask.shape == self.proj_size, \
                f"Expected mask of size {self.proj_size} but got " \
                f"{self.mask.shape} instead."
            assert self.device == self.mask.device, \
                f"Discrepancy in the devices of 'pos' and 'mask' attributes." \
                f" Please use `SameSettingImageData.to()` to set the device."

    def to_dict(self):
        return {key: getattr(self, key) for key in self._keys}

    @property
    def num_views(self):
        return self.pos.shape[0]

    @property
    def num_points(self):
        """
        Number of points implied by ImageMapping. Zero is 'mappings' is
        None.
        """
        return self.mappings.num_groups if self.mappings is not None else 0

    @property
    def img_size(self):
        """
        Current size of the 'x' and 'mappings'. Depends on the
        cropping size and the downsampling scale.
        """
        return tuple(int(x / self.downscale) for x in self.crop_size)

    @property
    def ref_size(self):
        """
        Initial size of the loaded image features and the mappings.

        This size is used as reference to characterize other
        SameSettingImageData attributes such as the crop offsets,
        resolution. As such, it should not be modified directly.
        """
        return self._ref_size

    @ref_size.setter
    def ref_size(self, ref_size):
        ref_size = tuple(ref_size)
        assert (self.x is None
                and self.mappings is None
                and self.mask is None) \
               or self.ref_size == ref_size, \
            "Can't directly edit 'ref_size' if 'x', 'mappings' and 'mask' " \
            "are not all None."
        assert len(ref_size) == 2, \
            f"Expected len(ref_size)=2 but got {len(ref_size)} instead."
        self._ref_size = ref_size

    @property
    def pixel_dtype(self):
        """
        Smallest torch dtype allowed by the resolution for encoding
        pixel coordinates.
        """
        for dtype in [torch.int16, torch.int32, torch.int64]:
            if torch.iinfo(dtype).max >= max(
                    self.ref_size[0], self.ref_size[1]):
                break
        return dtype

    @property
    def proj_upscale(self):
        """
        Upsampling scale factor of the projection map and mask size,
        with respect to 'ref_size'.

        Must follow: proj_upscale >= 1
        """
        return self._proj_upscale

    @proj_upscale.setter
    def proj_upscale(self, scale):
        assert (self.mask is None and self.mappings is None) \
               or self.proj_upscale == scale, \
            "Can't directly edit 'proj_upscale' if 'mask' and 'mappings' " \
            "are not both None."
        assert scale >= 1, \
            f"Expected scalar larger than 1 but got {scale} instead."
        # assert isinstance(scale, int), \
        #     f"Expected an int but got a {type(scale)} instead."
        # assert (scale & (scale-1) == 0) and scale != 0,\
        #     f"Expected a power of 2 but got {scale} instead."

        self._proj_upscale = scale

    @property
    def proj_size(self):
        """
        Size of the projection map and mask.

        This size is used to define the mask and initial mappings. It
        is defined by 'ref_size' and 'proj_upscale' and, as such, cannot
        be edited directly.
        """
        return tuple(int(x * self.proj_upscale) for x in self.ref_size)

    @property
    def rollings(self):
        """
        Rollings to apply to each image, with respect to the 'ref_size'
        state.

        By convention, rolling is applied first, then cropping, then
        resizing. For that reason, rollings should be defined before
        'x' or 'mappings' are cropped or resized.
        """
        return self._rollings

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
        """
        Update the rollings state of the SameSettingImageData, WITH
        RESPECT TO ITS REFERENCE STATE 'ref_size'.

        This assumes the images have a circular representation (ie that
        the first and last pixels along the width are adjacent in
        reality).

        Does not support prior cropping along the width or resizing.
        """
        # Make sure no prior cropping or resizing was applied to the
        # images and mappings
        assert self.ref_size[0] == self.img_size[0], \
            f"CenterRoll cannot operate if images and mappings " \
            f"underwent prior cropping or resizing."
        assert self.crop_size is None \
               or self.crop_size[0] == self.ref_size[0], \
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
            pix_roll = torch.repeat_interleave(
                self.rollings[self.mappings.images],
                self.mappings.values[1].pointers[1:]
                - self.mappings.values[1].pointers[:-1])

            # Recover the width pixel coordinates
            w_pix = self.mappings.pixels[:, 0].long()
            w_pix = (w_pix + pix_roll) % self.ref_size[0]
            w_pix = w_pix.type(self.pixel_dtype)

            # Apply pixel update
            self.mappings.pixels[:, 0] = w_pix
        
        return self

    @property
    def crop_size(self):
        """
        Size of the cropping to apply to the 'ref_size' to obtain the
        current image cropping.

        This size is used to characterize 'x' and 'mappings'. As
        such, it should not be modified directly.
        """
        return self._crop_size

    @crop_size.setter
    def crop_size(self, crop_size):
        crop_size = tuple(crop_size)
        assert (self.x is None and self.mappings is None) \
               or self.crop_size == crop_size, \
            "Can't directly edit 'crop_size' if 'x' or 'mappings' are " \
            "not both None. Consider using 'update_cropping'."
        assert len(crop_size) == 2, \
            f"Expected len(crop_size)=2 but got {len(crop_size)} instead."
        assert crop_size[0] <= self.ref_size[0] \
               and crop_size[1] <= self.ref_size[1], \
            f"Expected size smaller than {self.ref_size} but got " \
            f"{crop_size} instead."
        self._crop_size = crop_size

    @property
    def crop_offsets(self):
        """
        X-Y (width, height) offsets of the top-left corners of cropping
        boxes to apply to the 'ref_size' in order to obtain the current
        image cropping.

        These offsets must match the 'num_views' and is used to
        characterize 'x' and 'mappings'. As such, it should not be
        modified directly.
        """
        return self._crop_offsets

    @crop_offsets.setter
    def crop_offsets(self, crop_offsets):
        assert (self.x is None and self.mappings is None) \
               or (self.crop_offsets == crop_offsets).all(), \
            "Can't directly edit 'crop_offsets' if 'x' or 'mappings' " \
            "are not both None. Consider using 'update_cropping'."
        assert crop_offsets.dtype == torch.int64, \
            f"Expected dtype=torch.int64 but got dtype={crop_offsets.dtype} " \
            f"instead."
        assert crop_offsets.shape == (self.num_views, 2), \
            f"Expected tensor of shape {(self.num_views, 2)} but got " \
            f"{crop_offsets.shape} instead."
        self._crop_offsets = crop_offsets.to(self.device)

    def update_cropping(self, crop_size, crop_offsets):
        """
        Update the cropping state of the SameSettingImageData, WITH
        RESPECT TO ITS CURRENT STATE 'img_size'.

        Parameters crop_size and crop_offsets are resized to the
        'ref_size'

        Crop the 'x' and 'mappings', with respect to their current
        'img_size' (as opposed to the 'ref_size').
        """
        # Update the private 'crop_size' and 'crop_offsets' attributes
        # wrt 'ref_size'
        crop_offsets = crop_offsets.long()
        self._crop_size = tuple(int(x * self.downscale) for x in crop_size)
        self._crop_offsets = (self.crop_offsets
                              + crop_offsets * self.downscale).long()

        # Update the images' cropping
        #   - Image features have format: BxCxHxW
        #   - Crop size has format: (W, H)
        #   - Crop offsets have format: (W, H)
        if self.x is not None:
            x = [i[:, o[1]:o[1] + crop_size[1], o[0]:o[0] + crop_size[0]]
                 for i, o in zip(self.x, crop_offsets)]
            x = torch.cat([im.unsqueeze(0) for im in x])
            self.x = x

        # Update the mappings
        if self.mappings is not None:
            self.mappings = self.mappings.crop(crop_size, crop_offsets)
        
        return self

    @property
    def downscale(self):
        """
        Downsampling scale factor of the current image resolution, with
        respect to the initial image size 'ref_size'.

        Must follow: scale >= 1
        """
        return self._downscale

    @downscale.setter
    def downscale(self, scale):
        assert (self.x is None and self.mappings is None) \
               or self.downscale == scale, \
            "Can't directly edit 'downscale' if 'x' or 'mappings' are " \
            "not both None. Consider using 'update_features_and_scale'."
        assert scale >= 1, \
            f"Expected scalar larger than 1 but got {scale} instead."
        # assert isinstance(scale, int), \
        #     f"Expected an int but got a {type(scale)} instead."
        # assert (scale & (scale-1) == 0) and scale != 0,\
        #     f"Expected a power of 2 but got {scale} instead."
        self._downscale = scale

    def update_features_and_scale(self, x):
        """
        Update the downscaling state of the SameSettingImageData, WITH
        RESPECT TO ITS CURRENT STATE 'img_size'.

        Downscaling 'x' attribute is ambiguous. As such, they are
        expected to be scaled outside of the SameSettingImageData
        object before being passed to 'update_features_and_scale', for
        'downscale' and 'mappings' to be updated accordingly.
        """
        # Update internal attributes based on the input
        # downscaled image features
        scale = self.img_size[0] / x.shape[3]
        self._downscale = self.downscale * scale
        self.x = x

        if scale > 1:
            self.mappings = self.mappings.downscale_views(scale) \
                if self.mappings is not None else None
        
        return self

    @property
    def x(self):
        """
        Tensor of loaded image features with shape NxCxHxW, where
        N='num_views' and (W, H)='img_size'. Can be None if no image
        features were loaded.

        For clean load, consider using 'SameSettingImageData.load()'.
        """
        return self._x

    @x.setter
    def x(self, x):
        if x is None:
            self._x = None
        else:
            assert isinstance(x, torch.Tensor), \
                f"Expected a tensor of image features but got {type(x)} " \
                f"instead."
            assert x.shape[0] == self.num_views \
                   and x.shape[2:][::-1] == self.img_size, \
                f"Expected a tensor of shape ({self.num_views}, :, " \
                f"{self.img_size[1]}, {self.img_size[0]}) but got " \
                f"{x.shape} instead."
            self._x = x.to(self.device)

    @property
    def mappings(self):
        """
        ImageMapping data mapping 3D points to the images.

        The state of the mappings is closely linked to the state of the
        images. The image indices must agree with 'num_views', the
        pixel coordinates must correspond to the current 'img_size',
        scaling and cropping. As such, it is recommended not to
        manipulate the mappings directly.
        """
        return self._mappings

    @mappings.setter
    def mappings(self, mappings):
        if mappings is None:
            self._mappings = None
        else:
            assert isinstance(mappings, ImageMapping), \
                f"Expected an ImageMapping but got {type(mappings)} instead."
            unique_idx = torch.unique(mappings.images)
            img_idx = torch.arange(self.num_views)
            assert (unique_idx == img_idx).all(), \
                f"Image indices in the mappings do not match the " \
                f"SameSettingImageData image indices."
            w_max, h_max = mappings.pixels.max(dim=0).values
            assert w_max < self.img_size[0] and h_max < self.img_size[1], \
                f"Max pixel values should be smaller than ({self.img_size}) " \
                f"but got ({w_max, h_max}) instead."
            self._mappings = mappings.to(self.device)

    def select_points(self, idx, mode='pick'):
        """
        Update the 3D points sampling. Typically called after a 3D
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
            projection features) is present in the mapping, the value
            of one of the duplicates is picked at random. In this case,
            the correspondence map for the N points in the mapping must
            be provided as a 1D array of size N such that i -> idx[i].

        Returns a new SameSettingImageData object.
        """
        # TODO: careful with the device used at train time. Can't rely
        #  on CUDA...
        # TODO: make sure the merge mode works on real data...

        # Images are not affected if no mappings are present or idx is
        # None
        if self.mappings is None or idx is None:
            return self.clone()

        # Work on a clone of self, to avoid in-place modifications.
        idx = tensor_idx(idx)
        images = self.clone()

        # Picking mode by default
        if mode == 'pick':
            # Select mappings wrt the point index
            mappings = images.mappings.select_points(idx, mode=mode)

            # Select the images used in the mappings. Selected images
            # are sorted by their order in image_indices. Mappings'
            # image indices will also be updated to the new ones.
            # Mappings are temporarily removed from the images as they
            # will be affected by the indexing on images.
            seen_image_idx = lexunique(mappings.images)
            images.mappings = None
            images = images[seen_image_idx]
            images.mappings = mappings.select_views(seen_image_idx)

            return images

        # Merge mode
        elif mode == 'merge':
            assert idx.shape[0] == self.num_points, \
                f"Merge correspondences has size {idx.shape[0]} but size " \
                f"{self.num_points} was expected."
            assert (torch.arange(idx.max() + 1) == torch.unique(idx)).all(), \
                "Merge correspondences must map to a compact set of " \
                "indices."

            # Select mappings wrt the point index
            # Images are not modified, since the 'merge' mode
            # guarantees no image is discarded
            images.mappings = images.mappings.select_points(idx, mode=mode)

        else:
            raise ValueError(f"Unknown point selection mode '{mode}'.")

        return images

    @property
    def mask(self):
        """
        Boolean mask used for 3D points projection in the images.

        If not None, must be a BoolTensor of size 'proj_size'.
        """
        return self._mask

    @mask.setter
    def mask(self, mask):
        if mask is None:
            self._mask = None
        else:
            assert mask.dtype == torch.bool, \
                f"Expected a dtype=torch.bool but got dtype={mask.dtype} " \
                f"instead."
            assert mask.shape == self.proj_size, \
                f"Expected mask of size {self.proj_size} but got " \
                f"{mask.shape} instead."
            self._mask = mask.to(self.device)

    def load(self):
        """
        Load images to the 'x' attribute.

        Images are batched into a tensor of size NxCxHxW, where
        N='num_views' and (W, H)='img_size'. They are read with
        respect to their order in 'path', resized to 'ref_size', rolled
        with 'rollings', cropped with 'crop_size' and 'crop_offsets'
        and subsampled by 'downscale'.
        """
        self._x = self.read_images(
            size=self.ref_size,
            rollings=self.rollings,
            crop_size=self.crop_size,
            crop_offsets=self.crop_offsets,
            downscale=self.downscale).to(self.device)
        return self

    def read_images(self, idx=None, size=None, rollings=None, crop_size=None,
                    crop_offsets=None, downscale=None):
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
        if downscale is not None:
            assert downscale >= 1, \
                f"Expected scalar larger than 1 but got {downscale} instead."

        # Read images from files
        images = [Image.open(p).convert('RGB').resize(size)
                  for p in self.path[idx]]

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
        """
        Returns the number of image views in the SameSettingImageData.
        """
        return self.num_views

    def __getitem__(self, idx):
        """
        Indexing mechanism.

        Returns a new copy of the indexed SameSettingImageData.
        Supports torch and numpy indexing. For practical reasons, we
        don't want to have duplicate images in the SameSettingImageData,
        so indexing with duplicates
        will raise an error.
        """
        idx = tensor_idx(idx)
        assert torch.unique(idx).shape[0] == idx.shape[0], \
            f"Index must not contain duplicates."
        idx = idx.to(self.device)
        idx_numpy = np.asarray(idx.cpu())

        return self.__class__(
            path=self.path[idx_numpy],
            pos=self.pos[idx],
            opk=self.opk[idx],
            ref_size=copy.deepcopy(self.ref_size),
            proj_upscale=copy.deepcopy(self.proj_upscale),
            downscale=copy.deepcopy(self.downscale),
            crop_size=copy.deepcopy(self.crop_size),
            crop_offsets=self.crop_offsets[idx],
            voxel=copy.deepcopy(self.voxel),
            r_max=copy.deepcopy(self.r_max),
            r_min=copy.deepcopy(self.r_min),
            growth_k=copy.deepcopy(self.growth_k),
            growth_r=copy.deepcopy(self.growth_r),
            x=self.x[idx] if self.x is not None else None,
            mappings=self.mappings.select_views(idx)
            if self.mappings is not None else None,
            mask=self.mask if self.mask is not None else None)

    def __iter__(self):
        """
        Iteration mechanism.
        
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
        """
        Returns a shallow copy of self, except for 'x' and 'mappings',
        which are cloned as they may carry gradients.
        """
        out = copy.copy(self)
        out._x = self.x.clone() if self.x is not None \
            else None
        out._mappings = self.mappings.clone() if self.mappings is not None \
            else None
        return out

    def to(self, device):
        """Set torch.Tensor attributes device."""
        self.pos = self.pos.to(device)
        self.opk = self.opk.to(device)
        self._crop_offsets = self.crop_offsets.to(device)
        self._x = self.x.to(device) if self.x is not None \
            else None
        self._mappings = self.mappings.to(device) if self.mappings is not None \
            else None
        self._mask = self.mask.to(device) if self.mask is not None \
            else None
        return self

    @property
    def device(self):
        """Get the device of the torch.Tensor attributes."""
        return self.pos.device

    @property
    def settings_hash(self):
        """
        Produces a hash of the shared SameSettingImageData settings
        (except for the mask). This hash can be used as an identifier
        to characterize the SameSettingImageData for Batching
        mechanisms.
        """
        # Assert shared keys are the same for all items
        keys = tuple(set(SameSettingImageData._shared_keys)
                     - set([SameSettingImageData._mask_key]))
        return hash(tuple(getattr(self, k) for k in keys))

    @staticmethod
    def get_batch_type():
        """Required by MMData.from_mm_data_list."""
        return SameSettingImageBatch

    @property
    def feature_map_indexing(self):
        """
        Return the indices for extracting mapped data from the
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
        """
        Return the indices that will be used for atomic-level pooling on
        CSR-formatted data.
        """
        if self.mappings is not None:
            return self.mappings.atomic_csr_indexing
        return None

    @property
    def view_csr_indexing(self):
        """
        Return the indices that will be used for view-level pooling on
        CSR-formatted data.
        """
        if self.mappings is not None:
            return self.mappings.view_csr_indexing
        return None


class SameSettingImageBatch(SameSettingImageData):
    """
    Wrapper class of SameSettingImageData to build a batch from a list
    of SameSettingImageData and reconstruct it afterwards.

    Each SameSettingImageData in the batch is assumed to refer to
    different Data objects in its mappings. For that reason, if the
    SameSettingImageData have mappings, they will also be batched with
    their point ids reindexed. For consistency, this implies that
    associated Data points are expected to be batched in the same order.
    """

    def __init__(self, **kwargs):
        super(SameSettingImageBatch, self).__init__(**kwargs)
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
        # attributes, except for the 'mask' attribute, for which the
        # value of the first SameSettingImageData is taken for the
        # whole batch. This is because masks may differ slightly when
        # computed statistically with NonStaticImageMask.
        if len(image_data_list) > 1:

            # Make sure shared keys are the same across the batch
            hash_ref = image_data_list[0].settings_hash
            assert all(im.settings_hash == hash_ref
                       for im in image_data_list), \
                f"All SameSettingImageData values for shared keys " \
                f"{SameSettingImageData._shared_keys} must be the same " \
                f"(except for the 'mask')."

            for image_data in image_data_list[1:]:

                # Prepare stack keys for concatenation or batching
                image_dict = image_data.to_dict()
                for key, value in [(k, v) for (k, v) in image_dict.items()
                                   if k in SameSettingImageData._own_keys]:
                    batch_dict[key] += [value]

                # Prepare the sizes for items recovery with
                # .to_data_list
                sizes.append(image_data.num_views)

        # Concatenate numpy array attributes
        for key in SameSettingImageData._numpy_keys:
            batch_dict[key] = np.concatenate(batch_dict[key])

        # Concatenate torch array attributes
        for key in SameSettingImageData._torch_keys:
            batch_dict[key] = torch.cat(batch_dict[key])

        # Concatenate images, unless one of the items does not have
        # image features
        if any(img is None for img in batch_dict[SameSettingImageData._x_key]):
            batch_dict[SameSettingImageData._x_key] = None
        else:
            batch_dict[SameSettingImageData._x_key] = torch.cat(
                batch_dict[SameSettingImageData._x_key])

        # Batch mappings, unless one of the items does not have mappings
        if any(mpg is None
               for mpg in batch_dict[SameSettingImageData._map_key]):
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
    """
    Holder for SameSettingImageData items. Useful when
    SameSettingImageData can't be batched together because their
    internal settings differ. Default format for handling image
    attributes, features and mappings in multimodal models and modules.
    """

    def __init__(self, image_list: List[SameSettingImageData]):
        self._list = image_list
        self.debug()

    @property
    def num_settings(self):
        return len(self)

    @property
    def num_views(self):
        return sum([im.num_views for im in self])

    @property
    def num_points(self):
        return self[0].num_points

    @property
    def x(self):
        return [im.x for im in self]

    @x.setter
    def x(self, x_list):
        assert x_list is None or isinstance(x_list, list) \
               and all(isinstance(x, torch.Tensor) for x in x_list), \
            f"Expected a List(torch.Tensor) but got {type(x_list)} instead."

        if x_list is None:
            x_list = [None] * self.num_settings

        for im, x in (self, x_list):
            im.x = x

    def debug(self):
        assert isinstance(self._list, list), \
            f"Expected a list of SameSettingImageData but got " \
            f"{type(self._list)} instead."
        assert all(isinstance(im, SameSettingImageData) for im in self), \
            f"All list elements must be of type SameSettingImageData."
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
        assert isinstance(idx, int)
        assert idx < self.__len__()
        return self._list[idx]

    def __iter__(self):
        for i in range(self.__len__()):
            yield self[i]

    def __repr__(self):
        return f"{self.__class__.__name__}(num_settings={self.num_settings}, " \
               f"num_views={self.num_views}, num_points={self.num_points}, " \
               f"device={self.device})"

    def select_points(self, idx, mode='pick'):
        return self.__class__([im.select_points(idx, mode=mode)
                               for im in self])

    def update_features_and_scale(self, x_list):
        assert isinstance(x_list, list) \
               and all(isinstance(x, torch.Tensor) for x in x_list), \
            f"Expected a List(torch.Tensor) but got {type(x_list)} instead."
        self._list = [im.update_features_and_scale(x)
                      for im, x in zip(self, x_list)]
        return self

    def load(self):
        self._list = [im.load() for im in self]
        return self

    def clone(self):
        return self.__class__([im.clone() for im in self])

    def to(self, device):
        self._list = [im.to(device) for im in self]
        return self

    @property
    def device(self):
        return self[0].device

    @staticmethod
    def get_batch_type():
        """Required by MMData.from_mm_data_list."""
        return ImageBatch

    @property
    def feature_map_indexing(self):
        """
        Return the indices for extracting mapped data from the
        corresponding batch of image feature maps.

        The batch of image feature maps X is expected to have the shape
        `[B, C, H, W]`. The returned indexing object idx is intended to
        be used for recovering the mapped features as: `X[idx]`.
        """
        return [im.feature_map_indexing for im in self]

    @property
    def atomic_csr_indexing(self):
        """
        Return the indices that will be used for atomic-level pooling on
        CSR-formatted data.
        """
        return [im.atomic_csr_indexing for im in self]

    @property
    def view_cat_sorting(self):
        """
        Return the sorting indices to arrange concatenated view-level
        features to a CSR-friendly order wrt to points.
        """
        # Recover the expanded view idx for each SameSettingImageData
        # in self
        dense_idx_list = [
            torch.repeat_interleave(
                torch.arange(im.num_points),
                im.view_csr_indexing[1:] - im.view_csr_indexing[:-1])
            for im in self]

        # Assuming the corresponding view features will be concatenated
        # in the same order as in self, compute the sorting indices to
        # arrange features wrt point indices, to facilitate CSR indexing
        sorting = torch.cat([idx for idx in dense_idx_list]).argsort()

        return sorting

    @property
    def view_cat_csr_indexing(self):
        """
        Return the indices that will be used for view-level pooling on
        CSR-formatted data. To sort concatenated view-level features,
        see 'view_cat_sorting'.
        """
        # Assuming the features have been concatenated and sorted as
        # aforementioned in 'view_cat_sorting' compute the new CSR
        # indices to be used for feature view-pooling
        view_csr_idx = torch.cat([
            im.view_csr_indexing.unsqueeze(dim=1)
            for im in self], dim=1).sum(dim=1)
        return view_csr_idx


class ImageBatch(ImageData):
    """
    Wrapper class of ImageData to build a batch from a list
    of ImageData and reconstruct it afterwards.

    Like SameSettingImageBatch, each ImageData in the batch here is
    assumed to refer to different Data objects. Hence, the point ids in
    ImageBatch mappings are reindexed. For consistency, this
    implies that associated Data points are expected to be batched in
    the same sorder.
    """

    def __init__(self, image_list: List[SameSettingImageData]):
        super(ImageBatch, self).__init__(image_list)
        self.__il_sizes__ = None
        self.__hashes__ = None
        self.__il_idx_dict__ = None
        self.__im_idx_dict__ = None
        self.__cum_pts__ = None

    @staticmethod
    def from_data_list(image_data_list):
        assert isinstance(image_data_list, list) and len(image_data_list) > 0
        assert all(isinstance(x, ImageData)
                   for x in image_data_list)

        # Recover the list of unique hashes
        hashes = list(set([im.settings_hash
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

        # Number of points in the SameSettingImageData mappings
        n_pts_dict = {h: [] for h in hashes}

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
                    [torch.arange(cum_pts[il_idx], cum_pts[il_idx+1])
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
                im.mappings = im.mappings[self.__cum_pts__[il_idx]
                                          :self.__cum_pts__[il_idx+1]]

                # Update the list of MultiSettingImages with each
                # SameSettingImageData in its original position
                msi_list[il_idx][im_idx] = im

        # Convert to MultiSettingImage
        return [ImageData(x) for x in msi_list]


class ImageMapping(CSRData):
    """CSRData format for point-image-pixel mappings."""

    # TODO: expand to optional projection features in the view-level mappings

    @staticmethod
    def from_dense(point_ids, image_ids, pixels, num_points=None):
        """
        Recommended method for building an ImageMapping from dense data.
        """
        assert point_ids.ndim == 1, \
            'point_ids and image_ids must be 1D tensors'
        assert point_ids.shape == image_ids.shape, \
            'point_ids and image_ids must have the same shape'
        assert point_ids.shape[0] == pixels.shape[0], \
            'pixels and indices must have the same shape'

        # Sort by point_ids first, image_ids second
        idx_sort = lexargsort(point_ids, image_ids)
        image_ids = image_ids[idx_sort]
        point_ids = point_ids[idx_sort]
        pixels = pixels[idx_sort]
        del idx_sort

        # Convert to "nested CSRData" format.
        # Compute point-image pointers in the pixels array.
        # NB: The pointers are marked by non-successive point-image ids.
        #     Watch out for overflow in case the point_ids and
        #     image_ids are too large and stored in 32 bits.
        composite_ids = CompositeTensor(point_ids, image_ids)
        image_pixel_mappings = CSRData(composite_ids.data, pixels, dense=True)
        del composite_ids

        # Compress point_ids and image_ids by taking the last value of
        # each pointer
        image_ids = image_ids[image_pixel_mappings.pointers[1:] - 1]
        point_ids = point_ids[image_pixel_mappings.pointers[1:] - 1]

        # Instantiate the main CSRData object
        # Compute point pointers in the image_ids array
        mapping = ImageMapping(
            point_ids, image_ids, image_pixel_mappings, dense=True,
            is_index_value=[True, False])

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
        mapping = mapping.insert_empty_groups(point_ids,
                                              num_groups=num_points)

        return mapping

    def debug(self):
        # CSRData debug
        super(ImageMapping, self).debug()

        # ImageMapping-specific debug
        # TODO: change here to account for projection features
        assert len(self.values) == 2, \
            f"CSRData format does not match that of ImageMapping: " \
            f"len(values) should be 2 but is {len(self.values)}."
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
        return torch.arange(self.num_groups)

    @property
    def images(self):
        return self.values[0]

    @images.setter
    def images(self, images):
        self.values[0] = images.to(self.device)

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
        """
        Return the (w_min, w_max, h_min, hmax) pixel values per image.
        """
        # TODO: handle circular panoramic images and relevant cropping
        image_ids = torch.repeat_interleave(
            self.images,
            self.values[1].pointers[1:] - self.values[1].pointers[:-1])
        min_pix, _ = torch_scatter.scatter_min(self.pixels, image_ids, dim=0)
        max_pix, _ = torch_scatter.scatter_max(self.pixels, image_ids, dim=0)
        return min_pix[:, 0], max_pix[:, 0], min_pix[:, 1], max_pix[:, 1]

    @property
    def feature_map_indexing(self):
        """
        Return the indices for extracting mapped data from the
        corresponding batch of image feature maps.

        The batch of image feature maps X is expected to have the shape
        `[B, C, H, W]`. The returned indexing object idx is intended to
        be used for recovering the mapped features as: `X[idx]`.
        """
        idx_batch = torch.repeat_interleave(
            self.images,
            self.values[1].pointers[1:] - self.values[1].pointers[:-1])
        idx_height = self.pixels[:, 1]
        idx_width = self.pixels[:, 0]
        idx = (idx_batch.long(), ..., idx_height.long(), idx_width.long())
        return idx

    @property
    def atomic_csr_indexing(self):
        """
        Return the indices that will be used for atomic-level pooling on
        CSR-formatted data.
        """
        return self.values[1].pointers

    @property
    def view_csr_indexing(self):
        """
        Return the indices that will be used for view-level pooling on
        CSR-formatted data.
        """
        return self.pointers

    def downscale_views(self, ratio):
        """
        Update the image resolution after subsampling. Typically called
        after a pooling layer in an image CNN encoder.

        To update the image resolution in the mappings, the pixel
        coordinates are converted to lower resolutions. This operation
        is likely to produce duplicates. Searching and removing these
        duplicates only affects the atomic-level mappings, so only the
        pixel-level nested CSRData is modified by this function.

        Returns a new ImageMapping object.
        """
        # TODO: careful with the device used at train time. Can't rely
        #  on CUDA...
        assert ratio >= 1, \
            f"Invalid image subsampling ratio: {ratio}. Must be larger than 1."

        # Create a copy of self
        out = self.clone()

        # Expand atomic-level mappings to 'dense' format
        ids = torch.repeat_interleave(
            torch.arange(out.values[1].num_groups),
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
                ids,
                torch.stack((pix_x, pix_y), dim=1).type(pix_dtype),
                dense=True
            )
            out.values[1].__sizes__ = sizes
        elif isinstance(out.values[1], CSRData):
            out.values[1] = CSRData(
                ids,
                torch.stack((pix_x, pix_y), dim=1).type(pix_dtype),
                dense=True
            )
        else:
            raise NotImplementedError(
                "The atomic-level mappings must be either a CSRData or "
                "CSRBatch object.")

        return out

    def select_views(self, idx):
        """
        Return a copy of self with images selected with idx.

        Idx is assumed to refer to image indices. The mappings are
        updated so as to remove mappings to image indices absent from
        idx and change the image indexing to respect the new order
        implied by idx: idx[i] -> i.

        For the mappings to preserve their meaning, this operation
        assumes the same indexation is also applied to the
        corresponding SameSettingImageData and contains no duplicate
        indices.
        """
        idx = tensor_idx(idx)
        assert torch.unique(idx).shape[0] == idx.shape[0], \
            f"Index must not contain duplicates."
        idx = idx.to(self.device)

        # Get view-level indices for images to keep
        view_idx = torch.where((self.images[..., None] == idx).any(-1))[0]
        out = self.clone()

        # Index the values
        out.values = [val[view_idx] for val in out.values]

        # Update the image indices. To do so, create a tensor of indices
        # idx_gen so that the desired output can be computed with simple
        # indexation idx_gen[images]. This avoids using map() or
        # numpy.vectorize alternatives
        idx_gen = torch.full((idx.max() + 1,), -1, dtype=torch.int64)
        idx_gen = idx_gen.scatter_(0, idx, torch.arange(idx.shape[0]))
        out.images = idx_gen[out.images]

        # Update the pointers
        point_ids = torch.repeat_interleave(
            torch.arange(out.num_groups), out.pointers[1:] - out.pointers[:-1])
        point_ids = point_ids[view_idx]
        out.pointers = CSRData._sorted_indices_to_pointers(point_ids)

        # Some points may have been seen by no image so we need to
        # inject 0-sized pointers to account for these. To get the real
        # point_ids take the last value of each pointer
        point_ids = point_ids[out.pointers[1:] - 1]
        out = out.insert_empty_groups(point_ids, num_groups=self.num_groups)

        out.debug()

        return out

    def select_points(self, idx, mode='pick'):
        """
        Update the 3D points sampling. Typically called after a 3D
        sampling or strided convolution layer in a 3D CNN encoder.

        To update the 3D resolution in the modality data and mappings,
        two methods - ruled by the `mode` parameter - may be used:
        picking or merging.

          - 'pick' (default): only a subset of points is sampled, the
            rest is discarded. In this case, a 1D indexing array must be
            provided.

          - 'merge': points are agglomerated. The mappings are combined,
            duplicates are removed. If any other value (such as
            projection features) is present in the mapping, the value
            of one of the duplicates is picked at random. In this case,
            the correspondence map for the N points in the mapping must
            be provided as a 1D array of size N.

        Returns a new ImageMapping object.
        """
        # TODO: careful with the device used at train time. Can't rely
        #  on CUDA...
        # TODO: make sure the merge mode works on real data...
        MODES = ['pick', 'merge']
        assert mode in MODES, \
            f"Unknown mode '{mode}'. Supported modes are {MODES}."

        # Mappings are not affected if idx is None
        if idx is None:
            return self.clone()

        idx = tensor_idx(idx)

        # Picking mode by default
        if mode == 'pick':
            out = self[idx]

        # Merge mode
        elif mode == 'merge':
            assert idx.shape[0] == self.num_groups, \
                f"Merge correspondences has size {idx.shape[0]} but size " \
                f"{self.num_groups} was expected."
            assert (torch.arange(idx.max()+1) == torch.unique(idx)).all(), \
                "Merge correspondences must map to a compact set of " \
                "indices."

            # Expand to dense format
            point_ids = torch.repeat_interleave(
                idx, self.pointers[1:] - self.pointers[:-1])
            point_ids = torch.repeat_interleave(
                point_ids,
                self.values[1].pointers[1:] - self.values[1].pointers[:-1])
            image_ids = torch.repeat_interleave(
                self.images,
                self.values[1].pointers[1:] - self.values[1].pointers[:-1])
            pixels = self.pixels

            # Remove duplicates and aggregate projection features
            idx_unique = lexargunique(point_ids, image_ids, pixels[:, 0],
                                      pixels[:, 1])
            point_ids = point_ids[idx_unique]
            image_ids = image_ids[idx_unique]
            pixels = pixels[idx_unique]

            # Convert to CSR format
            out = ImageMapping.from_dense(point_ids, image_ids, pixels,
                                          num_points=idx.max()+1)
        else:
            raise ValueError(f"Unknown point selection mode '{mode}'.")

        return out

    def crop(self, crop_size, crop_offsets):
        """
        Return a copy of self with cropped image mappings.

        The mappings are updated so as to change pixel coordinates to
        account for a cropping of the mapped images. Each image has its
        own cropping offset, but all share the same cropping box size.

        Pixels discarded by the cropping will also be discarded from
        the mapping.

        For the mappings to preserve their meaning, this operation
        assumes the same cropping is also applied to the corresponding
        SameSettingImageData.
        """
        # TODO: expand to handle projection features here too
        assert crop_offsets.shape == (torch.unique(self.images).shape[0], 2), \
            f"Expected crop_offsets to have shape " \
            f"{(torch.unique(self.images).shape[0], 2)} but got shape " \
            f"{crop_offsets.shape} instead."

        # Distribute the offsets to the pixels
        #   - Crop offsets have format: (W, H)
        #   - Pixels have format: (W, H)
        image_ids = torch.repeat_interleave(
            self.images,
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
        point_ids = torch.repeat_interleave(
            torch.arange(self.num_groups),
            self.pointers[1:] - self.pointers[:-1])
        point_ids = torch.repeat_interleave(
            point_ids,
            self.values[1].pointers[1:] - self.values[1].pointers[:-1])
        image_ids = torch.repeat_interleave(
            self.images,
            self.values[1].pointers[1:] - self.values[1].pointers[:-1])

        # Select only the valid mappings and create a mapping
        point_ids = point_ids[cropped_in_idx]
        image_ids = image_ids[cropped_in_idx]
        pixels = pixels[cropped_in_idx]

        # Convert to CSR format
        return ImageMapping.from_dense(point_ids, image_ids, pixels,
                                       num_points=self.num_groups)


class ImageMappingBatch(ImageMapping, CSRBatch):
    """Batch wrapper for ImageMapping."""
    pass


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

idx, img_idx = lexsort(idx, img_idx)

m = ImageMapping.from_dense(idx, img_idx, pixels)

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
