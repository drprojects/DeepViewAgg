import copy
import numpy as np
from PIL import Image
import torch
from torch_points3d.datasets.multimodal.csr import CSRData, CSRDataBatch
from torch_points3d.utils.multimodal import composite_operation


# TODO Hold loaded images in the ImageData ?
# TODO if so, will ImageBatch must recover the proper indices ?

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

    _keys = ['path', 'pos', 'opk', 'img_size', 'mask', 'map_size_high', 'map_size_low', 'crop_top',
             'crop_bottom', 'voxel', 'r_max', 'r_min', 'growth_k', 'growth_r']
    _numpy_keys = ['path']
    _torch_keys = ['pos', 'opk']
    _array_keys = _numpy_keys + _torch_keys
    _shared_keys = list(set(_keys) - set(_array_keys))

    def __init__(self, path=np.empty(0, dtype='O'), pos=torch.empty([0, 3]), opk=torch.empty([0, 3]),
                 mask=None, img_size=(2048, 1024), map_size_high=(2048, 1024), map_size_low=(512, 256),
                 crop_top=0, crop_bottom=0, voxel=0.1, r_max=30, r_min=0.5, growth_k=0.2, growth_r=10,
                 **kwargs
                 ):

        assert path.shape[0] == pos.shape[0] and path.shape[0] == opk.shape[0], (f"Attributes ",
                                                                                 "'path', 'pos' and 'opk' must have the same length.")

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

    def to_dict(self):
        return {key: getattr(self, key) for key in self._keys}

    @property
    def num_images(self):
        return self.pos.shape[0]

    @property
    def map_size_low(self):
        return self.__map_size_low__

    @map_size_low.setter
    def map_size_low(self, map_size_low):
        self.__map_size_low__ = map_size_low

        # Update the optimal xy mapping dtype allowed by the resolution
        for dtype in [torch.uint8, torch.int16, torch.int32, torch.int64]:
            if torch.iinfo(dtype).max >= max(self.map_size_low[0], self.map_size_low[1]):
                break
        self.map_dtype = dtype

    def read_images(self, idx=None, size=None):
        """Read images and batch them into a tensor of size BxCxHxW."""
        if idx is None:
            idx = np.arange(self.num_images)
        if isinstance(idx, int):
            idx = np.array([idx])
        if isinstance(idx, torch.Tensor):
            idx = np.asarray(idx)
        if len(idx.shape) < 1:
            idx = np.array([idx])
        if size is None:
            size = self.img_size

        return torch.from_numpy(np.stack([
            np.array(Image.open(path).convert('RGB').resize(size, Image.LANCZOS))
            for path in self.path[idx]
        ])).permute(0, 3, 1, 2)

    def coarsen_coordinates(self, pixel_coordinates):
        """Convert pixel coordinates from high to low resolution."""
        ratio = self.map_size_low[0] / self.map_size_high[0]
        return torch.floor(torch.Tensor(pixel_coordinates) * ratio).type(self.map_dtype)

    def non_static_pixel_mask(self, size=None, n_sample=5):
        """
        Find the mask of identical pixels accross a list of images.
        """
        if size is None:
            size = self.map_size_high

        mask = torch.ones(size, dtype=torch.bool)

        n_sample = min(n_sample, self.num_images)
        if n_sample < 2:
            return mask

        # Iteratively update the mask w.r.t. a reference image
        idx = torch.multinomial(torch.arange(self.num_images, dtype=torch.float), n_sample)
        img_1 = self.read_images(idx=idx[0], size=size).squeeze()
        for i in idx[1:]:
            img_2 = self.read_images(idx=i, size=size).squeeze()
            mask_equal = torch.all(img_1 == img_2, axis=0).t()
            mask[torch.logical_and(mask, mask_equal)] = 0

        return mask

    def clone(self):
        """Returns a copy of the instance."""
        return self[np.arange(len(self))]

    def __len__(self):
        """Returns the number of images present."""
        return self.num_images

    def __getitem__(self, idx):
        """
        Indexing mechanism.

        Returns a new copy of the indexed ImadeData. Supports torch and numpy 
        indexing.
        """
        if isinstance(idx, int):
            idx = np.array([idx])

        if isinstance(idx, slice):
            idx_torch = idx
            idx_numpy = idx
        else:
            if isinstance(idx, np.ndarray):
                idx_torch = torch.from_numpy(idx)
                idx_numpy = idx
            elif isinstance(idx, torch.Tensor):
                idx_torch = idx
                idx_numpy = np.asarray(idx)
            else:
                raise NotImplementedError

        return self.__class__(
            path=self.path[idx_numpy].copy(),
            pos=self.pos[idx_torch].clone(),
            opk=self.opk[idx_torch].clone(),
            mask=self.mask.clone() if self.mask is not None else self.mask,
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

    def __iter__(self):
        """
        Iteration mechanism.
        
        Looping over the ImageData will return an ImageData for each individual
        item.
        """
        for i in range(self.__len__()):
            yield self[i]

    def __repr__(self):
        return f"{self.__class__.__name__}(num_images={self.num_images}, device={self.device})"

    def to(self, device):
        """Set torch.Tensor attribute device."""
        self.pos = self.pos.to(device)
        self.opk = self.opk.to(device)
        self.mask = self.mask.to(device) if self.mask is not None else self.mask
        return self

    @property
    def device(self):
        """Get the device of the torch.Tensor attributes."""
        assert self.pos.device == self.opk.device, (f"Discrepancy in the devices of 'pos' and ",
                                                    "'opk' attributes. Please use `ImageData.to()` to set the device.")
        if self.mask is not None:
            assert self.pos.device == self.mask.device, (f"Discrepancy in the devices of 'pos' ",
                                                         "and 'mask' attributes. Please use `ImageData.to()` to set the device.")
        return self.pos.device


class ImageBatch(ImageData):
    """
    Wrapper class of ImageData to build a batch from a list of ImageData and 
    reconstruct it afterwards.  
    """

    def __init__(self, **kwargs):
        super(ImageBatch, self).__init__(**kwargs)
        self.__sizes__ = None

    @property
    def batch_jumps(self):
        return np.cumsum(np.concatenate(([0], self.__sizes__))) if self.__sizes__ is not None \
            else None

    @property
    def batch_items_sizes(self):
        return self.__sizes__ if self.__sizes__ is not None else None

    @property
    def num_batch_items(self):
        return len(self.__sizes__) if self.__sizes__ is not None else None

    @staticmethod
    def from_image_data_list(image_data_list):
        assert isinstance(image_data_list, list) and len(image_data_list) > 0

        # Recover the attributes of the first ImageData to compare the shared
        # attributes with the other ImageData
        batch_dict = image_data_list[0].to_dict()
        sizes = [image_data_list[0].num_images]
        for key in ImageData._array_keys:
            batch_dict[key] = [batch_dict[key]]

        # Only stack if all ImageData have the same shared attributes, except
        # for the 'mask' attribute, for which the value of the first ImageData
        # is taken for the whole batch. This is because masks may differ
        # slightly when computed statistically with NonStaticImageMask.
        if len(image_data_list) > 1:
            for image_data in image_data_list[1:]:

                image_dict = image_data.to_dict()

                for key, value in [(k, v) for (k, v) in image_dict.items()
                                   if k in ImageData._shared_keys]:
                    if key != 'mask':
                        assert batch_dict[key] == value, \
                            (f"All ImageData values for shared keys {ImageData._shared_keys} ",
                             "must be the same (except for the 'mask').")

                for key, value in [(k, v) for (k, v) in image_dict.items()
                                   if k in ImageData._array_keys]:
                    batch_dict[key] += [value]
                sizes.append(image_data.num_images)

        # Concatenate array attributes with torch or numpy
        for key in ImageData._numpy_keys:
            batch_dict[key] = np.concatenate(batch_dict[key])

        for key in ImageData._torch_keys:
            batch_dict[key] = torch.cat(batch_dict[key])

        # Initialize the batch from dict and keep track of the item sizes
        batch = ImageBatch(**batch_dict)
        batch.__sizes__ = np.array(sizes)

        return batch

    def to_image_data_list(self):
        if self.__sizes__ is None:
            raise RuntimeError(('Cannot reconstruct image data list from batch because the batch ',
                                'object was not created using `ImageBatch.from_image_data_list()`.'))

        batch_jumps = self.batch_jumps
        return [self[batch_jumps[i]:batch_jumps[i + 1]] for i in range(self.num_batch_items)]


class ImageMapping(CSRData):
    def __init__(self, point_ids, image_ids, pixels, num_points=None):
        assert point_ids.ndim == 1, \
            'point_ids and image_ids must be 1D tensors'
        assert point_ids.shape == image_ids.shape, \
            'point_ids and image_ids must have the same shape'
        assert point_ids.shape[0] == pixels.shape[0], \
            'pixels and indices must have the same shape'

        # Sort by point_ids first, image_ids second
        sorting, composite_ids = composite_operation(point_ids, image_ids,
            op='argsort', torch_out=True)
        image_ids = image_ids[sorting]
        point_ids = point_ids[sorting]
        pixels = pixels[sorting]
        del sorting

        # Convert to "nested CSRData" format.
        # Compute point-image jumps in the pixels array.
        # NB: The jumps are marked by non-successive point-image ids. Watch
        #     out for overflow in case the point_ids and image_ids are too
        #     large and stored in 32 bits.
        image_pixel_mappings = CSRData(composite_ids, pixels, dense=True)
        del composite_ids

        # Compress point_ids and image_ids by taking the last value of each
        # jump
        image_ids = image_ids[image_pixel_mappings.jumps[1:] - 1]
        point_ids = point_ids[image_pixel_mappings.jumps[1:] - 1]

        # Instantiate the main CSRData object
        # Compute point jumps in the image_ids array
        super(ImageMapping, self).__init__(point_ids, image_ids, image_pixel_mappings,
            dense=True, is_index_value=[True, False])

        # Some points may have been seen by no image so we need to inject
        # 0-sized jumps to account for these.
        # NB: we assume all relevant points are present in range(num_points),
        #     if a point with an id larger than num_points were to exist, we
        #     would not be able to take it into account in the jumps.
        if num_points is None or num_points < point_ids.max() + 1:
            num_points = point_ids.max() + 1

        # Compress point_ids by taking the last value of each jump
        point_ids = point_ids[self.jumps[1:] - 1]
        self._insert_empty_groups(point_ids, num_groups=num_points)

    @property
    def points(self):
        return torch.arange(self.num_groups)

    @property
    def images(self):
        return self.values[0]

    @property
    def pixels(self):
        return self.values[1].values[0]

    def get_batch_features_indexing(self):
        """
        Return indices for extracting unit-level data from image features
        batch.

        When the image features batch X is expected to have the shape
        [B, C, W, H], the returned indices idx_1, idx_2, idx_3 are intended to
        be used so that X[idx_1, :, idx_2, idx_3] are the unit-level features
        of the mapping.
        """
        idx_batch = torch.repeat_interleave(
            self.images,
            self.values[1].jumps[1:] - self.values[1].jumps[:-1]
        )
        idx_width = self.pixels[0]
        idx_height = self.pixels[1]
        return idx_batch, idx_width, idx_height

    def get_unit_pooling_indices(self):
        """
        Return the indices that will be used for unit-level pooling.
        """
        raise NotImplementedError

    def get_unit_pooling_csr_indices(self):
        """
        Return the indices that will be used for unit-level pooling on
        CSR-formatted data.
        """
        return self.values[1].jumps

    def get_view_pooling_indices(self):
        """
        Return the indices that will be used for view-level pooling.
        """
        raise NotImplementedError

    def get_view_pooling_csr_indices(self):
        """
        Return the indices that will be used for view-level pooling on
        CSR-formatted data.
        """
        return self.jumps

    # TODO: update mappings after image pooling
    # TODO: update mappings after 3D sampling. Case1: sampling. Case2: voxel pooling

    def reduce_2d_resolution(self, ratio):
        """
        Update the image resolution after subsampling. Typically called after
        a pooling layer in an image CNN encoder.
        """
        assert ratio >= 1, \
            f"Invalid image resampling ratio: {ratio}. Must be larger than 1."

        # expand all mappings to 'dense' format

        # floor divide pixels

        # search duplicates

        # Update unit-level and view-level mappings
        # keep the first instances only
        # Assuming this does not cause issues for other potential unit-level CSR-nested values

        #



    def reduce_3d_resolution(self, idx, volumetric=False):
        pass