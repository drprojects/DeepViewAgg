import copy
import numpy as np
from PIL import Image
import torch
from torch_points3d.datasets.multimodal.csr import CSRData, CSRDataBatch
from torch_points3d.utils.multimodal import lexargsort, lexargunique, \
    CompositeTensor



class ImageData(object):
    """
    Class to hold arrays of images information, along with shared 3D-2D 
    projection information.

    Attributes
        path            numpy.ndarray     image paths
        pos             torch.Tensor      image position
        opk             torch.Tensor      image angular poses

        img_size        tuple             RGB image size (width, height)
        mask            torch.BoolTensor  projection mask
        map_size_high   tuple             projection map high resolution size
        map_size_low    tuple             projection map low resolution size
        crop_top        int               projection map top crop pixels
        crop_bottom     int               projection map bottom crop pixels
        voxel           float             projection voxel resolution
        r_max           float             projection radius max
        r_min           float             projection radius min
        growth_k        float             projection pixel growth factor
        growth_r        float             projection pixel growth radius
    """

    _keys = [
        'path', 'pos', 'opk', 'images', 'img_size', 'mask', 'map_size_high',
        'map_size_low', 'crop_top', 'crop_bottom', 'voxel', 'r_max', 'r_min',
        'growth_k', 'growth_r']
    _numpy_keys = ['path']
    _torch_keys = ['pos', 'opk', 'images']
    _array_keys = _numpy_keys + _torch_keys
    _shared_keys = list(set(_keys) - set(_array_keys))

    def __init__(
            self, path=np.empty(0, dtype='O'), pos=torch.empty([0, 3]),
            opk=torch.empty([0, 3]), mask=None, img_size=(2048, 1024),
            map_size_high=(2048, 1024), map_size_low=(512, 256), crop_top=0,
            crop_bottom=0, voxel=0.1, r_max=30, r_min=0.5, growth_k=0.2,
            growth_r=10, **kwargs):

        assert path.shape[0] == pos.shape[0] and path.shape[0] == opk.shape[0],\
            f"Attributes 'path', 'pos' and 'opk' must have the same length."

        self.path = np.array(path)
        self.pos = pos.double()
        self.opk = opk.double()
        self.img_size = tuple(img_size)
        self.mask = mask
        self.map_size_high = tuple(map_size_high)
        self.map_size_low = tuple(map_size_low)
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
        self.voxel = voxel
        self.r_max = r_max
        self.r_min = r_min
        self.growth_k = growth_k
        self.growth_r = growth_r
        self._images = None

    def to_dict(self):
        return {key: getattr(self, key) for key in self._keys}

    @property
    def num_images(self):
        return self.pos.shape[0]

    @property
    def map_size_low(self):
        return self._map_size_low

    @map_size_low.setter
    def map_size_low(self, map_size_low):
        self._map_size_low = map_size_low

        # Update the optimal xy mapping dtype allowed by the resolution
        for dtype in [torch.uint8, torch.int16, torch.int32, torch.int64]:
            if torch.iinfo(dtype).max >= max(
                    self.map_size_low[0], self.map_size_low[1]):
                break
        self.map_dtype = dtype
        
    @property
    def images(self):
        return self._images

    def load(self, size=None):
        """
        Read images and batch them into a tensor of size BxCxHxW.
        Images are then stored in the `images` attribute.
        """
        self._images = self.read_images(size=size).to(self.device)

    def read_images(self, idx=None, size=None):
        """Read images and batch them into a tensor of size BxCxHxW."""
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
        if size is None:
            size = self.img_size

        return torch.from_numpy(np.stack([
            np.array(Image.open(p).convert('RGB').resize(size, Image.LANCZOS))
            for p in self.path[idx]])).permute(0, 3, 1, 2)

    def non_static_pixel_mask(self, size=None, n_sample=5):
        """
        Find the mask of identical pixels across a list of images.
        """
        if size is None:
            size = self.map_size_high

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
        numpy indexing.
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
        idx = idx.to(self.device)
        idx_numpy = np.asarray(idx)

        out = self.__class__(
            path=self.path[idx_numpy].copy(),
            pos=self.pos[idx].clone(),
            opk=self.opk[idx].clone(),
            mask=self.mask.clone() if self.mask is not None else None,
            img_size=copy.deepcopy(self.img_size),
            map_size_high=copy.deepcopy(self.map_size_high),
            map_size_low=copy.deepcopy(self.map_size_low),
            crop_top=copy.deepcopy(self.crop_top),
            crop_bottom=copy.deepcopy(self.crop_bottom),
            voxel=copy.deepcopy(self.voxel),
            r_max=copy.deepcopy(self.r_max),
            r_min=copy.deepcopy(self.r_min),
            growth_k=copy.deepcopy(self.growth_k),
            growth_r=copy.deepcopy(self.growth_r)
        )

        out._images = self.images[idx].clone() if self.images is not None \
            else None

        return out

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
        """Set torch.Tensor attribute device."""
        self.pos = self.pos.to(device)
        self.opk = self.opk.to(device)
        self.mask = self.mask.to(device) if self.mask is not None \
            else self.mask
        self._images = self.images.to(device) if self.images is not None \
            else self.images
        return self

    @property
    def device(self):
        """Get the device of the torch.Tensor attributes."""
        assert self.pos.device == self.opk.device, \
            f"Discrepancy in the devices of 'pos' and 'opk' attributes. " \
            f"Please use `ImageData.to()` to set the device."
        if self.mask is not None:
            assert self.pos.device == self.mask.device, \
                f"Discrepancy in the devices of 'pos' and 'mask' attributes." \
                f" Please use `ImageData.to()` to set the device."
        if self.images is not None:
            assert self.pos.device == self.images.device, \
                f"Discrepancy in the devices of 'pos' and 'images' " \
                f"attributes. Please use `ImageData.to()` to set the device."
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
        for key in ImageData._array_keys:
            batch_dict[key] = [batch_dict[key]]

        # Only stack if all ImageData have the same shared attributes,
        # except for the 'mask' attribute, for which the value of the
        # first ImageData is taken for the whole batch. This is because
        # masks may differ slightly when computed statistically with
        # NonStaticImageMask.
        if len(image_data_list) > 1:
            for image_data in image_data_list[1:]:

                image_dict = image_data.to_dict()

                for key, value in [(k, v) for (k, v) in image_dict.items()
                                   if k in ImageData._shared_keys]:
                    if key != 'mask':
                        assert batch_dict[key] == value, \
                            f"All ImageData values for shared keys " \
                            f"{ImageData._shared_keys} must be the " \
                            f"same (except for the 'mask')."

                for key, value in [(k, v) for (k, v) in image_dict.items()
                                   if k in ImageData._array_keys]:
                    # Only stack if all ImageData have either not loaded
                    # their images or all

                    batch_dict[key] += [value]
                sizes.append(image_data.num_images)

        # Concatenate array attributes with torch or numpy
        for key in ImageData._numpy_keys:
            batch_dict[key] = np.concatenate(batch_dict[key])

        for key in ImageData._torch_keys:
            if key == 'images' and any(img is None for img in batch_dict[key]):
                batch_dict[key] = None
                continue
            batch_dict[key] = torch.cat(batch_dict[key])

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

    # TODO: padding circular affects coordinates, beware of mappings, beware of mappings validity
    # TODO: mask and crop affects coordinates, beware of mappings validity
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
        assert idx.shape[0] == self.num_groups,\
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
