import copy
import numpy as np
from PIL import Image
import torch
import torch_scatter
from torch_points3d.datasets.multimodal.csr import CSRData, CSRDataBatch
from torch_points3d.utils.multimodal import lexargsort, lexargunique, \
    CompositeTensor


class ImageData(object):
    """
    Class to hold arrays of images information, along with shared 3D-2D 
    projection information.

    Attributes
        path:numpy.ndarray     image paths
        pos:torch.Tensor      image position
        opk:torch.Tensor      image angular poses

        ref_size:tuple       initial size of the loaded images and the mappings
        proj_upscale
        crop_size
        crop_offsets

        voxel:float             projection voxel resolution
        r_max:float             projection radius max
        r_min:float             projection radius min
        growth_k:float             projection pixel growth factor
        growth_r:float             projection pixel growth radius

        images
        mappings
        mask            torch.BoolTensor  projection mask
    """

    _numpy_keys = ['path']
    _torch_keys = ['pos', 'opk', 'crop_offsets']
    _map_key = 'mappings'
    _img_key = 'images'
    _mask_key = 'mask'
    _shared_keys = ['ref_size', 'proj_upscale', 'crop_size', 'voxel', 'r_max',
                    'r_min', 'growth_k', 'growth_r'] + [_mask_key]
    _own_keys = _numpy_keys + _torch_keys + [_map_key, _img_key]
    _keys = _shared_keys + _own_keys

    def __init__(
            self,
            path=np.empty(0, dtype='O'),
            pos=torch.empty([0, 3]),
            opk=torch.empty([0, 3]),
            ref_size=(512, 256),
            proj_upscale=2,
            crop_size=None,
            crop_offsets=None,
            voxel=0.1,
            r_max=30,
            r_min=0.5,
            growth_k=0.2,
            growth_r=10,
            images=None,
            mappings=None,
            mask=None,
            **kwargs):

        assert path.shape[0] == pos.shape[0] == opk.shape[0], \
            f"Attributes 'path', 'pos' and 'opk' must have the same length."

        self.path = np.array(path)
        self.pos = pos.double()
        self.opk = opk.double()
        self.ref_size = ref_size
        self.proj_upscale = proj_upscale
        self.crop_size = crop_size if crop_size is not None else self.ref_size
        self.crop_offsets = crop_offsets if crop_offsets is not None \
            else torch.zeros((self.num_images, 2), dtype=torch.int64)
        self.downscale = 1
        self.voxel = voxel
        self.r_max = r_max
        self.r_min = r_min
        self.growth_k = growth_k
        self.growth_r = growth_r
        self.images = images
        self.mappings = mappings
        self.mask = mask

        self.debug()

        # TODO: create a crop 2D function

        # TODO: create an ImageMapping pixel info function

        # TODO: create a subsampling 2D function

        # TODO: initial circular custom padding? This would affect all sizes,
        #  mappings, crop offsets, etc. For now, let's just leave that aside.

    def debug(self):
        assert self.path.shape[0] == self.pos.shape[0] == self.opk.shape[0], \
            f"Attributes 'path', 'pos' and 'opk' must have the same length."
        assert self.pos.device == self.opk.device, \
            f"Discrepancy in the devices of 'pos' and 'opk' attributes. " \
            f"Please use `ImageData.to()` to set the device."
        assert len(tuple(self.ref_size)) == 2, \
            f"Expected len(ref_size)=2 but got {len(self.ref_size)} instead."
        assert self.proj_upscale >= 1, \
            f"Expected scalar larger than 1 but got {self.proj_upscale} " \
            f"instead."
        assert len(tuple(self.crop_size)) == 2, \
            f"Expected len(crop_size)=2 but got {len(self.crop_size)} instead."
        assert all(a <= b for a, b in zip(self.crop_size, self.ref_size)), \
            f"Expected size smaller than {self.ref_size} but got " \
            f"{self.crop_size} instead."
        assert self.crop_offsets.shape == (self.num_images, 2), \
            f"Expected tensor of shape {(self.num_images, 2)} but got " \
            f"{self.crop_offsets.shape} instead."
        assert self._downscale >= 1, \
            f"Expected scalar larger than 1 but got {self._downscale} instead."

        if self.images is not None:
            assert isinstance(self.images, torch.Tensor), \
                f"Expected a tensor of images but got {type(self.images)} " \
                f"instead."
            assert self.images.shape[0] == self.num_images \
                   and self.images.shape[2] == self.img_size[1] \
                   and self.images.shape[3] == self.img_size[0], \
                f"Expected a tensor of shape ({self.num_images}, :, " \
                f"{self.img_size[1]}, {self.img_size[0]}) but got " \
                f"{self.images.shape} instead."
            assert self.device == self.images.device, \
                f"Discrepancy in the devices of 'pos' and 'images' " \
                f"attributes. Please use `ImageData.to()` to set the device."

        if self.mappings is not None:
            assert isinstance(self.mappings, ImageMapping), \
                f"Expected an ImageMapping but got {type(self.mappings)} " \
                f"instead."
            unique_idx = torch.unique(self.mappings.images)
            img_idx = torch.arange(self.num_images)
            assert (unique_idx == img_idx).all(), \
                f"Image indices in the mappings do not match the ImageData " \
                f"image indices."
            w_max, h_max = self.mappings.pixels.max(dim=0).values
            assert w_max < self.img_size[0] and h_max < self.img_size[1], \
                f"Max pixel values should be smaller than ({self.img_size}) " \
                f"but got ({w_max, h_max}) instead."
            assert self.device == self.mappings.device, \
                f"Discrepancy in the devices of 'pos' and 'mappings' " \
                f"attributes. Please use `ImageData.to()` to set the device."
            self.mappings.debug()

        if self.mask is not None:
            assert isinstance(self.mask, torch.BoolTensor), \
                f"Expected a BoolTensor but got {type(self.mask)} instead."
            assert self.mask.shape == self.proj_size, \
                f"Expected mask of size {self.proj_size} but got " \
                f"{self.mask.shape} instead."
            assert self.device == self.mask.device, \
                f"Discrepancy in the devices of 'pos' and 'mask' attributes." \
                f" Please use `ImageData.to()` to set the device."

    def to_dict(self):
        return {key: getattr(self, key) for key in self._keys}

    @property
    def num_images(self):
        return self.pos.shape[0]

    @property
    def ref_size(self):
        """
        Initial size of the loaded images and the mappings.

        This size is used as reference to characterize other ImageData
        attributes such as the crop offsets, resolution. As such, it
        should not be modified directly.
        """
        return self._ref_size

    @ref_size.setter
    def ref_size(self, ref_size):
        assert self.images is None \
               and self.mappings is None \
               and self.mask is None, \
            "Can't set 'ref_size' if either 'images', 'mappings' or 'mask' " \
            "is not None."
        assert len(tuple(ref_size)) == 2, \
            f"Expected len(ref_size)=2 but got {len(ref_size)} instead."
        self._ref_size = tuple(ref_size)

    @property
    def pixel_dtype(self):
        """
        Smallest torch dtype allowed by the resolution for encoding
        pixel coordinates.
        """
        for dtype in [torch.uint8, torch.int16, torch.int32, torch.int64]:
            if torch.iinfo(dtype).max >= max(
                    self.ref_size[0], self.ref_size[1]):
                break
        return dtype

    @property
    def proj_upscale(self):
        """
        Upsampling scale factor of the projection map and mask size, with
        respect to 'ref_size'.

        Must follow: proj_upscale >= 1
        """
        return self._proj_upscale

    @proj_upscale.setter
    def proj_upscale(self, scale):
        assert scale >= 1, \
            f"Expected scalar larger than 1 but got {scale} instead."
        # assert isinstance(scale, int), \
        #     f"Expected an int but got a {type(scale)} instead."
        # assert (scale & (scale-1) == 0) and scale != 0,\
        #     f"Expected a power of 2 but got {scale} instead."
        assert self.mask is None and self.mappings is None, \
            "Can't set 'proj_upscale' if 'mask' and 'mappings' are not both " \
            "None."
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
    def crop_size(self):
        """
        Size of the cropping to apply to the 'ref_size' to obtain the
        current image cropping.

        This size is used to characterize 'images' and 'mappings'. As
        such, it should not be modified directly.
        """
        return self._crop_size

    @crop_size.setter
    def crop_size(self, crop_size):
        assert self.images is None and self.mappings is None, \
            "Can't set 'crop_size' if 'images' or 'mappings' are not both None."
        assert len(tuple(crop_size)) == 2, \
            f"Expected len(crop_size)=2 but got {len(crop_size)} instead."
        assert crop_size[0] <= self.ref_size[0] \
               and crop_size[1] <= self.ref_size[1], \
            f"Expected size smaller than {self.ref_size} but got {crop_size} " \
            f"instead."
        self._crop_size = tuple(crop_size)

    @property
    def crop_offsets(self):
        """
        X-Y offsets of the top-left corners of cropping boxes to apply
        to the 'ref_size' in order to obtain the current image cropping.

        These offsets must match the 'num_images' and is used to
        characterize 'images' and 'mappings'. As such, it should not be
        modified directly.
        """
        return self._crop_offsets

    @crop_offsets.setter
    def crop_offsets(self, crop_offsets):
        assert self.images is None and self.mappings is None, \
            "Can't set 'crop_size' if 'images' or 'mappings' are not both None."
        assert isinstance(crop_offsets, torch.LongTensor), \
            f"Expected LongTensor but got {type(crop_offsets)} instead."
        assert crop_offsets.shape == (self.num_images, 2), \
            f"Expected tensor of shape {(self.num_images, 2)} but got " \
            f"{crop_offsets.shape} instead."
        self._crop_offsets = crop_offsets.to(self.device)

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
        assert scale >= 1, \
            f"Expected scalar larger than 1 but got {scale} instead."
        # assert isinstance(scale, int), \
        #     f"Expected an int but got a {type(scale)} instead."
        # assert (scale & (scale-1) == 0) and scale != 0,\
        #     f"Expected a power of 2 but got {scale} instead."
        self._downscale = scale

    @property
    def img_size(self):
        """
        Current size of the 'images' and 'mappings'. Depends on the
        cropping size and the downsampling scale.
        """
        return tuple(int(x / self.downscale) for x in self.crop_size)

    @property
    def images(self):
        """
        Tensor of loaded images with shape NxCxHxW, where N='num_images'
        and (W, H)='img_size'. Can be None if no images were loaded.

        For clean load, consider using 'ImageData.load_images()'.
        """
        return self._images

    @images.setter
    def images(self, images):
        if images is not None:
            assert isinstance(images, torch.Tensor), \
                f"Expected a tensor of images but got {type(images)} instead."
            assert images.shape[0] == self.num_images \
                   and images.shape[2] == self.img_size[1] \
                   and images.shape[3] == self.img_size[0], \
                f"Expected a tensor of shape ({self.num_images}, :, " \
                f"{self.img_size[1]}, {self.img_size[0]}) but got " \
                f"{images.shape} instead."
        self._images = images.to(self.device)

    @property
    def mappings(self):
        """
        ImageMapping data mapping 3D points to the images.

        The state of the mappings is closely linked to the state of the
        images. The image indices must agree with 'num_images', the
        pixel coordinates must correspond to the current 'img_size',
        scaling and cropping. As such, it is recommended not to
        manipulate the mappings directly.
        """
        return self._mappings

    @mappings.setter
    def mappings(self, mappings):
        if mappings is not None:
            assert isinstance(mappings, ImageMapping), \
                f"Expected an ImageMapping but got {type(mappings)} instead."
            unique_idx = torch.unique(mappings.images)
            img_idx = torch.arange(self.num_images)
            assert (unique_idx == img_idx).all(), \
                f"Image indices in the mappings do not match the ImageData " \
                f"image indices."
            w_max, h_max = mappings.pixels.max(dim=0).values
            assert w_max < self.img_size[0] and h_max < self.img_size[1], \
                f"Max pixel values should be smaller than ({self.img_size}) " \
                f"but got ({w_max, h_max}) instead."
        self._mappings = mappings.to(self.device)

    @property
    def mask(self):
        """
        Boolean mask used for 3D points projection in the images.

        If not None, must be a BoolTensor of size 'proj_size'.
        """
        return self._mask

    @mask.setter
    def mask(self, mask):
        if mask is not None:
            assert isinstance(mask, torch.BoolTensor), \
                f"Expected a BoolTensor but got {type(mask)} instead."
            assert mask.shape == self.proj_size, \
                f"Expected mask of size {self.proj_size} but got " \
                f"{mask.shape} instead."
        self._mask = mask.to(self.device)

    def load_images(self):
        """
        Load images to the 'images' attribute.

        Images are batched into a tensor of size NxCxHxW, where
        N='num_images' and (W, H)='img_size'. They are read with
        respect to their order in 'path', resized to 'ref_size',
        cropped with 'crop_size' and 'crop_offsets' and subsampled by
        'downscale'.
        """
        self._images = self.read_images(
            size=self.ref_size,
            crop_size=self.crop_size,
            crop_offsets=self.crop_offsets,
            downscale=self.downscale).to(self.device)

    def read_images(self, idx=None, size=None, crop_size=None,
                    crop_offsets=None, downscale=None):
        # TODO: faster read with multiprocessing:
        #  https: // stackoverflow.com / questions / 19695249 / load - just - part - of - an - image - in -python
        #  https://towardsdatascience.com/10x-faster-parallel-python-without-python-multiprocessing-e5017c93cce1
        """
        Read images and batch them into a tensor of size BxCxHxW.

        Images are indexed with 'idx' with respect to their order in
        'path', then resized to 'size', before being cropped with
        'crop_size' and 'crop_offsets' and subsampled by 'downscale'.
        """
        # Index to select part of the images in 'path'
        if idx is None:
            idx = np.arange(self.num_images)
        elif isinstance(idx, int):
            idx = np.array([idx])
        elif isinstance(idx, torch.Tensor):
            idx = np.asarray(idx)
        elif isinstance(idx, slice):
            idx = np.arange(self.num_images)[idx]
        if len(idx.shape) < 1:
            idx = np.array([idx])

        # Size to which the images should be reshaped
        if size is None:
            size = self.img_size

        # Cropping boxes size and offsets
        # XAND(crop_size and crop_offsets)
        assert bool(crop_size) == bool(crop_offsets), \
            f"If either 'crop_size' or 'crop_offsets' is specified, both " \
            f"must be specified."
        if crop_size is not None:
            crop_size = tuple(crop_size)
            assert len(crop_size) == 2, \
                f"Expected len(crop_size)=2 but got {len(crop_size)} instead."
            assert all(a <= b for a, b in zip(crop_size, size)), \
                f"Expected crop_size to be smaller than size but got size={size} " \
                f"and crop_size={crop_size} instead."
            assert isinstance(crop_offsets, torch.LongTensor), \
                f"Expected LongTensor but got {type(crop_offsets)} instead."
            assert crop_offsets.shape == (idx.shape[0], 2), \
                f"Expected tensor of shape {(idx.shape[0], 2)} but got " \
                f"{crop_offsets.shape} instead."
        else:
            crop_size = size
            crop_offsets = torch.zeros((idx.shape[0], 2), dtype=torch.int64)

        # Downsampling after cropping
        if downscale is not None:
            assert downscale >= 1, \
                f"Expected scalar larger than 1 but got {downscale} instead."

        # Read images from files
        images = [Image.open(p).convert('RGB').resize(size)
                  for p in self.path[idx]]

        # Crop and resize
        if downscale is None:
            w, h = crop_size
            images = [im.crop((left, top, left + w, top + h))
                      for im, (left, top) in zip(images, np.asarray(crop_offsets))]
        else:
            end_size = tuple(int(x / downscale) for x in crop_size)
            w, h = crop_size
            images = [im.resize(end_size, box=(left, top, left + w, top + h))
                      for im, (left, top) in zip(images, np.asarray(crop_offsets))]

        # Convert to torch batch
        images = torch.from_numpy(np.stack([np.asarray(im) for im in images]))
        images = images.permute(0, 3, 1, 2)

        return images

    def non_static_pixel_mask(self, size=None, n_sample=5):
        # TODO: make internal function, affecting the internal state, like
        #  load, crop, subsample
        """
        Find the mask of identical pixels across a list of images.
        """
        if size is None:
            size = self.proj_size

        mask = torch.ones(size, dtype=torch.bool)

        n_sample = min(n_sample, self.num_images)
        if n_sample < 2:
            return mask

        # Iteratively update the mask w.r.t. a reference image
        idx = torch.multinomial(
            torch.arange(self.num_images, dtype=torch.float), n_sample)
        img_1 = self.read_images(idx=idx[0], size=size).squeeze()
        for i in idx[1:]:
            img_2 = self.read_images(idx=i, size=size).squeeze()
            mask_equal = torch.all(img_1 == img_2, axis=0).t()
            mask[torch.logical_and(mask, mask_equal)] = 0

        return mask

    def clone(self):
        """Returns a copy of the instance."""
        return self[torch.arange(len(self))]

    def __len__(self):
        """Returns the number of images present."""
        return self.num_images

    def __getitem__(self, idx):
        """
        Indexing mechanism.

        Returns a new copy of the indexed ImageData. Supports torch and
        numpy indexing. For practical reasons, we don't want to have
        duplicate images in the ImageData, so indexing with duplicates
        will raise an error.
        """
        if isinstance(idx, int):
            idx = torch.LongTensor([idx])
        elif isinstance(idx, list):
            idx = torch.LongTensor(idx)
        elif isinstance(idx, slice):
            idx = torch.arange(self.num_images)[idx]
        elif isinstance(idx, np.ndarray):
            idx = torch.from_numpy(idx)
        # elif not isinstance(idx, torch.LongTensor):
        #     raise NotImplementedError
        assert idx.dtype is torch.int64, \
            "ImageData only supports int and torch.LongTensor indexing."
        assert idx.shape[0] > 0, \
            "ImageData only supports non-empty indexing. At least one " \
            "index must be provided."
        assert torch.unique(idx).shape[0] == idx.shape[0], \
            f"Index must not contain duplicates."
        idx = idx.to(self.device)
        idx_numpy = np.asarray(idx)

        return self.__class__(
            path=self.path[idx_numpy].copy(),
            pos=self.pos[idx].clone(),
            opk=self.opk[idx].clone(),
            ref_size=copy.deepcopy(self.ref_size),
            proj_upscale=copy.deepcopy(self.proj_upscale),
            crop_size=copy.deepcopy(self.crop_size),
            crop_offsets=self.crop_offsets[idx].clone(),
            voxel=copy.deepcopy(self.voxel),
            r_max=copy.deepcopy(self.r_max),
            r_min=copy.deepcopy(self.r_min),
            growth_k=copy.deepcopy(self.growth_k),
            growth_r=copy.deepcopy(self.growth_r),
            images=self.images[idx].clone() if self.images else None,
            mappings=self.mappings.index_images(idx) if self.mappings else None,
            mask=self.mask.clone() if self.mask is not None else None)

    def __iter__(self):
        """
        Iteration mechanism.
        
        Looping over the ImageData will return an ImageData for each
        individual item.
        """
        for i in range(self.__len__()):
            yield self[i]

    def __repr__(self):
        return f"{self.__class__.__name__}(num_images={self.num_images}, " \
               f"device={self.device})"

    def to(self, device):
        """Set torch.Tensor attributes device."""
        self.pos = self.pos.to(device)
        self.opk = self.opk.to(device)
        self.crop_offsets = self.crop_offsets.to(device)
        self.images = self.images.to(device) if self.images is not None \
            else None
        self.mappings = self.mappings.to(device) if self.mappings is not None \
            else None
        self.mask = self.mask.to(device) if self.mask is not None \
            else None
        return self

    @property
    def device(self):
        """Get the device of the torch.Tensor attributes."""
        return self.pos.device


class ImageBatch(ImageData):
    """
    Wrapper class of ImageData to build a batch from a list of
    ImageData and reconstruct it afterwards.
    """

    def __init__(self, **kwargs):
        super(ImageBatch, self).__init__(**kwargs)
        self.__sizes__ = None

    @property
    def batch_jumps(self):
        return np.cumsum(np.concatenate(([0], self.__sizes__))) \
            if self.__sizes__ is not None else None

    @property
    def batch_items_sizes(self):
        return self.__sizes__ if self.__sizes__ is not None else None

    @property
    def num_batch_items(self):
        return len(self.__sizes__) if self.__sizes__ is not None else None

    @staticmethod
    def from_image_data_list(image_data_list):
        assert isinstance(image_data_list, list) and len(image_data_list) > 0
        assert all(isinstance(x, ImageData) for x in image_data_list)

        # Recover the attributes of the first ImageData to compare the
        # shared attributes with the other ImageData
        batch_dict = image_data_list[0].to_dict()
        sizes = [image_data_list[0].num_images]
        for key in ImageData._own_keys:
            batch_dict[key] = [batch_dict[key]]

        # Only stack if all ImageData have the same shared attributes,
        # except for the 'mask' attribute, for which the value of the
        # first ImageData is taken for the whole batch. This is because
        # masks may differ slightly when computed statistically with
        # NonStaticImageMask.
        if len(image_data_list) > 1:
            for image_data in image_data_list[1:]:

                image_dict = image_data.to_dict()

                # Assert shared keys are the same for all items
                for key, value in [(k, v) for (k, v) in image_dict.items()
                                   if k in ImageData._shared_keys]:
                    if key != ImageData._mask_key:
                        assert batch_dict[key] == value, \
                            f"All ImageData values for shared keys " \
                            f"{ImageData._shared_keys} must be the " \
                            f"same (except for the 'mask')."

                # Prepare stack keys for concatenation or batching
                for key, value in [(k, v) for (k, v) in image_dict.items()
                                   if k in ImageData._own_keys]:
                    batch_dict[key] += [value]

                # Prepare the sizes for items recovery with
                # .to_image_data_list
                sizes.append(image_data.num_images)

        # Concatenate numpy array attributes
        for key in ImageData._numpy_keys:
            batch_dict[key] = np.concatenate(batch_dict[key])

        # Concatenate torch array attributes
        for key in ImageData._torch_keys:
            batch_dict[key] = torch.cat(batch_dict[key])

        # Concatenate images, unless one of the items does not have
        # images
        if any(img is None for img in batch_dict[ImageData._img_key]):
            batch_dict[ImageData._img_key] = None
        else:
            batch_dict[ImageData._img_key] = torch.cat(
                batch_dict[ImageData._img_key])

        # Batch mappings, unless one of the items does not have mappings
        if any(mpg is None for mpg in batch_dict[ImageData._map_key]):
            batch_dict[ImageData._map_key] = None
        else:
            batch_dict[ImageData._map_key] = \
                ImageMappingBatch.from_csr_list(batch_dict[ImageData._map_key])

        # Initialize the batch from dict and keep track of the item
        # sizes
        batch = ImageBatch(**batch_dict)
        batch.__sizes__ = np.array(sizes)

        return batch

    def to_image_data_list(self):
        if self.__sizes__ is None:
            raise RuntimeError(
                'Cannot reconstruct image data list from batch because '
                'the batch object was not created using '
                '`ImageBatch.from_image_data_list()`.')

        batch_jumps = self.batch_jumps
        return [self[batch_jumps[i]:batch_jumps[i + 1]]
                for i in range(self.num_batch_items)]


class ImageDataHolder:
    # holds list / dict of ImageData with shape as key

    # debug:
    # all items are ImageData
    # all have the shape they are supposed to have (if not None)

    # load: all ImageData load
    # load with their own crop arguments
    # TODO: create a holder for multiple ImageData, to be used when
    #  manipualting different sizes, resolutions, etc
    def view_pooling_arangement_index(self):
        # Index to apply to concatenated unit-pooled features, so that the
        # features are ordered by point ID. This way, we can use a CSR
        # representation to view-pool them using scatter_csr.
        pass

    def view_pooling_csr_indices(self):
        # CSR jumps of the concatenated and re-aranged (with
        # view_pooling_arangement_index) unit-pooled features.
        pass


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
        # Compute point-image jumps in the pixels array.
        # NB: The jumps are marked by non-successive point-image ids.
        #     Watch out for overflow in case the point_ids and
        #     image_ids are too large and stored in 32 bits.
        composite_ids = CompositeTensor(point_ids, image_ids)
        image_pixel_mappings = CSRData(composite_ids.data, pixels, dense=True)
        del composite_ids

        # Compress point_ids and image_ids by taking the last value of
        # each jump
        image_ids = image_ids[image_pixel_mappings.jumps[1:] - 1]
        point_ids = point_ids[image_pixel_mappings.jumps[1:] - 1]

        # Instantiate the main CSRData object
        # Compute point jumps in the image_ids array
        mapping = ImageMapping(
            point_ids, image_ids, image_pixel_mappings, dense=True,
            is_index_value=[True, False])

        # Some points may have been seen by no image so we need to
        # inject 0-sized jumps to account for these.
        # NB: we assume all relevant points are present in
        # range(num_points), if a point with an id larger than
        # num_points were to exist, we would not be able to take it
        # into account in the jumps.
        if num_points is None or num_points < point_ids.max() + 1:
            num_points = point_ids.max() + 1

        # Compress point_ids by taking the last value of each jump
        point_ids = point_ids[mapping.jumps[1:] - 1]
        mapping._insert_empty_groups(point_ids, num_groups=num_points)

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
        self.values[0] = images
        self.debug()

    @property
    def pixels(self):
        return self.values[1].values[0]

    @pixels.setter
    def pixels(self, pixels):
        self.values[1].values[0] = pixels
        self.debug()

    @staticmethod
    def get_batch_type():
        """Required by CSRDataBatch.from_csr_list."""
        return ImageMappingBatch

    @property
    def image_stats(self):
        """
        Return the (w_min, w_max, h_min, hmax) pixel values per image.
        """
        min_pix, _ = torch_scatter.scatter_min(self.pixels, self.images, dim=0)
        max_pix, _ = torch_scatter.scatter_max(self.pixels, self.images, dim=0)
        return min_pix[:, 0], max_pix[:, 0], min_pix[:, 1], max_pix[:, 1]

    @property
    def feature_map_indexing(self):
        """
        Return the indices for extracting mapped data from the
        corresponding batch of image feature maps.

        The batch of image feature maps X is expected to have the shape
        `[B, C, H, W]`. The returned indices idx_1, idx_2, idx_3 are
        intended to be used for recovering the mapped features as:
        `X[idx_1, :, idx_2, idx_3]`.
        """
        idx_batch = torch.repeat_interleave(
            self.images,
            self.values[1].jumps[1:] - self.values[1].jumps[:-1])
        idx_height = self.pixels[:, 1]
        idx_width = self.pixels[:, 0]
        return idx_batch.long(), idx_height.long(), idx_width.long()

    @property
    def unit_pooling_indices(self):
        """
        Return the indices that will be used for unit-level pooling.
        """
        raise NotImplementedError

    # TODO: padding, mask and crop affects coordinates, beware of mappings validity
    @property
    def unit_pooling_csr_indices(self):
        """
        Return the indices that will be used for unit-level pooling on
        CSR-formatted data.
        """
        return self.values[1].jumps

    @property
    def view_pooling_indices(self):
        """
        Return the indices that will be used for view-level pooling.
        """
        raise NotImplementedError

    @property
    def view_pooling_csr_indices(self):
        """
        Return the indices that will be used for view-level pooling on
        CSR-formatted data.
        """
        return self.jumps

    def subsample_2d(self, ratio):
        """
        Update the image resolution after subsampling. Typically called
        after a pooling layer in an image CNN encoder.

        To update the image resolution in the mappings, the pixel
        coordinates are converted to lower resolutions. This operation
        is likely to produce duplicates. Searching and removing these
        duplicates only affects the unit-level mappings, so only the
        pixel-level nested CSRData is modified by this function.

        Returns a new ImageMapping object.
        """
        # TODO: Take the occlusions into account when reducing the
        #  resolution ? Is it problematic if a point that should have
        #  been occluded - had the projection been directly computed on
        #  the lower resolution, is visible ?
        # TODO: careful with the device used at train time. Can't rely
        #  on CUDA...
        assert ratio >= 1, \
            f"Invalid image subsampling ratio: {ratio}. Must be larger than 1."

        # Create a copy of self
        out = copy.deepcopy(self)

        # Expand unit-level mappings to 'dense' format
        ids = torch.repeat_interleave(
            torch.arange(out.values[1].num_groups),
            out.values[1].jumps[1:] - out.values[1].jumps[:-1])
        pix_x = out.values[1].values[0][:, 0]
        pix_y = out.values[1].values[0][:, 1]
        pix_dtype = pix_x.dtype

        # Convert pixel coordinates to new resolution
        pix_x = (pix_x // ratio).long()
        pix_y = (pix_y // ratio).long()

        # Remove duplicates and sort wrt ids
        # Assuming this does not cause issues for other potential
        # unit-level CSR-nested values
        idx_unique = lexargunique(ids, pix_x, pix_y)
        ids = ids[idx_unique]
        pix_x = pix_x[idx_unique]
        pix_y = pix_y[idx_unique]

        # Build the new unit-level CSR mapping
        if isinstance(out.values[1], CSRDataBatch):
            sizes = out.values[1].__sizes__
            out.values[1] = CSRDataBatch(
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
                "The unit-level mappings must be either a CSRData or "
                "CSRDataBatch object.")

        return out

    def subsample_3d(self, idx, merge=False):
        """
        Update the 3D resolution after subsampling. Typically called
        after a 3D sampling or sampling layer in a 3D CNN encoder.

        To update the 3D resolution in the mappings, two methods may
        be used picking and merging, ruled by the  `merge` parameter.

          - Picking (default): only a subset of points is sampled, the
            rest is discarded. In this case, a 1D indexing array must be
            provided.

          - Merging: points are agglomerated. The mappings are combined,
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
        assert isinstance(idx, torch.LongTensor)

        # Picking mode by default
        if not merge:
            return self[idx]

        # Merge mode
        assert idx.shape[0] == self.num_groups, \
            f"Merge correspondences has size {idx.shape[0]} but size " \
            f"{self.num_groups} was expected."
        assert torch.equal(torch.arange(idx.max()), torch.unique(idx)), \
            "Merge correspondences must map to a compact set of " \
            "indices."

        # Expand to dense format
        point_ids = torch.repeat_interleave(
            idx, self.jumps[1:] - self.jumps[:-1])
        point_ids = torch.repeat_interleave(
            point_ids, self.values[1].jumps[1:] - self.values[1].jumps[:-1])
        image_ids = torch.repeat_interleave(
            self.images,
            self.values[1].jumps[1:] - self.values[1].jumps[:-1])
        pixels = self.pixels

        # Remove duplicates and aggregate projection features
        idx_unique = lexargunique(point_ids, image_ids, pixels)
        point_ids = point_ids[idx_unique]
        image_ids = image_ids[idx_unique]
        pixels = pixels[idx_unique]

        # Convert to CSR format
        return ImageMapping.from_dense(point_ids, image_ids, pixels)

    def index_images(self, idx):
        """
        Return a copy of self with images selected with idx.

        Idx is assumed to refer to image indices. The mappings are
        updated so as to remove mappings to image indices absent from
        idx and change the image indexing to respect the new order
        implied by idx: idx[i] -> i.

        For the mappings to preserve their meaning, this operation
        assumes the same indexation is also applied to the
        corresponding ImageData and contains no duplicate indices.
        """
        if isinstance(idx, int):
            idx = torch.LongTensor([idx])
        elif isinstance(idx, list):
            idx = torch.LongTensor(idx)
        elif isinstance(idx, slice):
            idx = torch.arange(self.images.max())[idx]
        elif isinstance(idx, np.ndarray):
            idx = torch.from_numpy(idx)
        # elif not isinstance(idx, torch.LongTensor):
        #     raise NotImplementedError
        assert idx.dtype is torch.int64, \
            "index_images only supports int and torch.LongTensor indexing."
        assert idx.shape[0] > 0, \
            "index_images only supports non-empty indexing. At least one " \
            "index must be provided."
        assert torch.unique(idx).shape[0] == idx.shape[0], \
            f"Index must not contain duplicates."
        idx = idx.to(self.device)

        # Get view-level indices for images to keep
        view_idx = (self.images[..., None] == idx).any(-1).nonzero()
        out = self.clone()

        # Index the values
        out.values = [val[view_idx] for val in out.values]

        # Update the jumps
        jumps = torch.repeat_interleave(
            torch.arange(out.num_groups), out.jumps[1:] - out.jumps[:-1])
        jumps = CSRData._sorted_indices_to_jumps(jumps[view_idx])
        out.jumps = jumps

        # Update the image indices. To do so, create a tensor of indices
        # idx_gen so that the desired output can be computed with simple
        # indexation idx_gen[images]. This avoids using map() or
        # numpy.vectorize alternatives
        idx_gen = torch.full((idx.max() + 1,), -1, dtype=torch.int64)
        idx_gen = idx_gen.scatter_(0, idx, torch.arange(idx.shape[0]))
        out.images = idx_gen[out.images]

        return out


class ImageMappingBatch(ImageMapping, CSRDataBatch):
    """Batch wrapper for ImageMapping."""
    pass


"""

import torch
from torch_points3d.datasets.multimodal.csr import *
from torch_points3d.datasets.multimodal.image import *
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

print((b.to_csr_list()[2].jumps == m.jumps).all().item())
print((b.to_csr_list()[2].values[1].values[0] == m.values[1].values[0]).all().item())

b[[0,0,1]]

b = CSRDataBatch.from_csr_list([m[2], m[1:3], m, m[0]])

#-----------------------------------------------

jumps = torch.LongTensor([0, 0,  5, 12, 12, 15])
val = torch.arange(15)
m = CSRData(jumps, val, dense=False)
b = CSRDataBatch.from_csr_list([m, m, m])

# b[[0, 1, 7, 8, 14]]
b[[0,0,5]]

"""
