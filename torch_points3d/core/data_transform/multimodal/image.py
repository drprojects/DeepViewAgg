import numpy as np
import torch
from torch_geometric.data import Data
from torch_points3d.core.data_transform import SphereSampling
from torch_points3d.datasets.multimodal.image import ImageData, ImageMapping
from torch_points3d.utils.multimodal import lexunique
import torchvision.transforms as T
from .projection import compute_index_map
from tqdm.auto import tqdm as tq

"""
Image-based transforms for multimodal data processing. Inspired by 
torch_points3d and torch_geometric transforms on data, with a signature 
allowing for multimodal transform composition: 
__call(data, images, mappings)__
"""


class NonStaticImageMask:
    """
    Transform-like structure. Find the mask of identical pixels across
    a list of images.
    """

    def __init__(self, mask_size=(2048, 1024), n_sample=5):
        self.mask_size = tuple(mask_size)
        self.n_sample = n_sample

    def _process(self, images):
        images.map_size_high = self.mask_size
        images.mask = images.non_static_pixel_mask(
            size=self.mask_size, n_sample=self.n_sample)
        return images

    def __call__(self, data, images, mappings=None):
        """
        Compute the projection of data points into images and return
        the input data augmented with attributes mapping points to
        pixels in provided images.

        Expects Data (or anything), ImageData or List(ImageData), 
        CSRData mapping (or anything).

        Returns the same input. The mask is saved in ImageData
        attributes, to be used for any subsequent image processing.
        """
        if isinstance(images, list):
            images = [self._process(img) for img in images]
        else:
            images = self._process(images)
        return data, images, mappings

    def __repr__(self):
        return self.__class__.__name__


class PointImagePixelMapping:
    """
    Transform-like structure. Computes the mappings between individual
    3D points and image pixels. Point mappings are identified based on
    the self.key point identifiers.
    """

    def __init__(
            self,
            map_size_high=(2048, 1024),
            map_size_low=(512, 256),
            crop_top=0,
            crop_bottom=0,
            voxel=0.1,
            r_max=30,
            r_min=0.5,
            growth_k=0.2,
            growth_r=10,
            empty=0,
            no_id=-1,
            key='point_index'
    ):

        self.key = key
        self.empty = empty
        self.no_id = no_id

        # Store the projection parameters destined for the ImageData
        # attributes.
        self.map_size_high = tuple(map_size_high)
        self.map_size_low = tuple(map_size_low)
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
        self.voxel = voxel
        self.r_max = r_max
        self.r_min = r_min
        self.growth_k = growth_k
        self.growth_r = growth_r

    def _process(self, data, images):
        assert hasattr(data, self.key)
        assert isinstance(images, ImageData)
        assert images.num_images >= 1, \
            "At least one image must be provided."

        # Pass the projection attributes to the ImageData
        images.map_size_high = self.map_size_high
        images.map_size_low = self.map_size_low
        images.crop_top = self.crop_top
        images.crop_bottom = self.crop_bottom
        images.voxel = self.voxel
        images.r_max = self.r_max
        images.r_min = self.r_min
        images.growth_k = self.growth_k
        images.growth_r = self.growth_r

        if images.mask is not None:
            assert images.mask.shape == images.map_size_high

        # Initialize the mapping arrays
        image_ids = []
        point_ids = []
        pixels = []

        from time import time
        t_sphere_sampling = 0
        t_projection = 0
        t_init_torch_pixels = 0
        t_ratio_pixels = 0
        t_unique_pixels = 0
        t_stack_pixels = 0
        t_append = 0

        # Project each image and gather the point-pixel mappings
        for i_image, image in tq(enumerate(images)):
            # Subsample the surrounding point cloud
            start = time()
            sampler = SphereSampling(image.r_max, image.pos, align_origin=False)
            data_sample = sampler(data)
            # data_sample = sampler(data.clone())
            t_sphere_sampling += time() - start

            # Projection to build the index map
            start = time()
            id_map, _ = compute_index_map(
                (data_sample.pos - image.pos.squeeze()).numpy(),
                getattr(data_sample, self.key).numpy(),
                np.array(image.opk.squeeze()),
                img_mask=image.mask.numpy() if image.mask is not None else None,
                img_size=image.map_size_high,
                crop_top=image.crop_top,
                crop_bottom=image.crop_bottom,
                voxel=image.voxel,
                r_max=image.r_max,
                r_min=image.r_min,
                growth_k=image.growth_k,
                growth_r=image.growth_r,
                empty=self.empty,
                no_id=self.no_id,
            )
            t_projection += time() - start

            # Convert the id_map to id-xy coordinate soup
            # First column holds the point indices, subsequent columns
            # hold the  pixel coordinates. We use this heterogeneous
            # soup to search for duplicate rows after resolution
            # coarsening.
            # NB: no_id pixels are ignored
            start = time()
            id_map = torch.from_numpy(id_map)
            pix_x_, pix_y_ = torch.where(id_map != self.no_id)
            # point_ids_pixel_soup = id_map[active_pixels]
            # point_ids_pixel_soup = np.column_stack((point_ids_pixel_soup,
            #                                         torch.stack(active_pixels, dim=1).transpose()))
            point_ids_ = id_map[(pix_x_, pix_y_)]
            # pixels_ = torch.stack(active_pixels, dim=1)

            # Skip image if no mapping was found
            if point_ids_.shape[0] == 0:
                continue
            t_init_torch_pixels += time() - start

            # Convert to lower resolution coordinates
            # NB: we assume the resolution ratio is the same for both 
            # dimensions
            start = time()
            ratio = image.map_size_high[0] / image.map_size_low[0]
            pix_x_ = (pix_x_ // ratio).long()
            pix_y_ = (pix_y_ // ratio).long()
            t_ratio_pixels += time() - start
            # pixels_ = image.coarsen_coordinates(pixels_)
            # point_ids_pixel_soup[:, 1:] = image.coarsen_coordinates(point_ids_pixel_soup[:, 1:])

            # Remove duplicate id-xy in low resolution
            # Sort by point id
            # point_ids_pixel_soup = np.unique(point_ids_pixel_soup,
            #                                  axis=0)  # bottleneck here ! Custom unique-sort with numba ?
            start = time()
            point_ids_, pix_x_, pix_y_ = lexunique(point_ids_, pix_x_, pix_y_)
            t_unique_pixels += time() - start

            # Cast pixel coordinates to a dtype minimizing memory use
            # point_ids_ = point_ids_pixel_soup[:, 0]
            # pixels_ = np.asarray(torch.from_numpy(point_ids_pixel_soup[:, 1:]).type(image.map_dtype))
            start = time()
            pixels_ = torch.stack((pix_x_, pix_y_), dim=1).type(image.map_dtype)
            # del point_ids_pixel_soup
            t_stack_pixels += time() - start

            # Gather per-image mappings in list structures, only to be
            # torch-concatenated once all images are processed
            start = time()
            image_ids.append(i_image)
            point_ids.append(point_ids_)
            pixels.append(pixels_)
            del pixels_, point_ids_
            t_append += time() - start

        print(f"    Cumulated times")
        print(f"        t_sphere_sampling: {t_sphere_sampling:0.3f}")
        print(f"        t_projection: {t_projection:0.3f}")
        print(f"        t_init_torch_pixels: {t_init_torch_pixels:0.3f}")
        print(f"        t_ratio_pixels: {t_ratio_pixels:0.3f}")
        print(f"        t_unique_pixels: {t_unique_pixels:0.3f}")
        print(f"        t_stack_pixels: {t_stack_pixels:0.3f}")
        print(f"        t_append: {t_append:0.3f}")

        # Drop the KD-tree attribute saved in data by the SphereSampling
        delattr(data, SphereSampling.KDTREE_KEY)

        # Raise error if no point-image-pixel mapping was found
        image_ids = torch.LongTensor(image_ids)
        if image_ids.shape[0] == 0:
            raise ValueError(
                "No mappings were found between the 3D points and any "
                "of the provided images. This will cause errors in the "
                "subsequent operations. Make sure your images are "
                "located in the vicinity of your point cloud and that "
                "the projection parameters allow for at least one "
                "point-image-pixel mapping before re-running this "
                "transformation.")

        # Reindex seen images
        # We want all images present in the mappings and in ImageData to
        # have been seen. If an image has not been seen, we remove it
        # here.
        # NB: The reindexing here relies on the fact that `unique`
        #  values are expected to be returned sorted.
        # seen_image_ids = np.unique(image_ids)
        # images = images[np.isin(np.arange(images.num_images), seen_image_ids)]
        # image_ids = np.digitize(image_ids, seen_image_ids) - 1
        start = time()
        seen_image_ids = lexunique(image_ids)
        images = images[seen_image_ids]
        image_ids = torch.bucketize(image_ids, seen_image_ids)
        print(f"        t_index_image_data: {time() - start:0.3f}")

        # Concatenate mappings data
        # image_ids = np.repeat(image_ids, [x.shape[0] for x in point_ids])
        # point_ids = np.concatenate(point_ids)
        # pixels = np.vstack(pixels)
        start = time()
        image_ids = torch.repeat_interleave(
            image_ids, torch.LongTensor([x.shape[0] for x in point_ids]))
        point_ids = torch.cat(point_ids)
        pixels = torch.cat(pixels)
        print(f"        t_concat_dense_mappings_data: {time() - start:0.3f}")

        start = time()
        mappings = ImageMapping.from_dense(
            point_ids, image_ids, pixels,
            num_points=getattr(data, self.key).numpy().max() + 1)
        print(f"        t_ImageMapping_init: {time() - start:0.3f}\n")

        return data, images, mappings

    def __call__(self, data, images, mappings=None):
        """
        Compute the projection of data points into images and return
        the input data augmented with attributes mapping points to
        pixels in provided images.

        Expects a Data and a ImageData or a List(Data) and a
        List(ImageData) of matching lengths.

        Returns the input data and the point-image-pixel mappings in a
        nested CSRData format.
        """
        if isinstance(data, list):
            assert isinstance(images, list) and len(data) == len(images), \
                f"List(Data) items and List(ImageData) must have the same " \
                f"lengths."
            out = [self._process(d, i) for d, i in zip(data, images)]
            data, images, mappings = [list(x) for x in zip(*out)]

        else:
            data, images, mappings = self._process(data, images)

        return data, images, mappings

    def __repr__(self):
        return self.__class__.__name__


class DropPoorMappings:
    """
    Transform to drop images and corresponding mappings when mappings
    account for less than a given ratio of the image area.
    """
    def __init__(self, ratio=0.05):
        self.ratio = 0.05

    def __call__(self, data, images, mappings):
        # TODO: DropPoorMappings
        pass

    def __repr__(self):
        return self.__class__.__name__


class CropFromMask:
    """Transform to crop top and bottom from images and mappings based on mask."""
    # TODO: CropFromMask
    pass


class PadMappings:
    """Transform to update the mappings to account for image padding."""
    # TODO: PadMappings
    pass


class PointImagePixelMappingFromId:
    """
    Transform-like structure. Intended to be called on _datas and
    _images_datas.

    Populate the passed Data sample in-place with attributes extracted
    from the input CSRData mappings, based on the self.key point
    identifiers.
    
    The indices in data are expected to be included in those in
    mappings. The CSRData format implicitly holds values for all
    self.key in [0, ..., len(mappings)].
    """

    def __init__(self, key='point_index', keep_unseen_images=False):
        self.key = key
        self.keep_unseen_images = keep_unseen_images

    def _process(self, data, images, mappings):
        assert isinstance(data, Data)
        assert hasattr(data, self.key)
        assert isinstance(images, ImageData)
        assert isinstance(mappings, ImageMapping)

        # Point indices to subselect mappings.
        # The selected mappings are sorted by their order in
        # point_indices.
        # NB: just like images, the same point may be used multiple
        # times.
        mappings = mappings[data[self.key]]

        # Update point indices to the new mappings length.
        # This is important to preserve the mappings and for multimodal
        # data batching mechanisms.
        data[self.key] = torch.arange(data.num_nodes)

        # If unseen images must still be kept
        # May be useful in case we want to keep track of global images
        if self.keep_unseen_images:
            return data, images, mappings

        # Subselect the images used in the mappings. The selected
        # images are sorted by their order in image_indices.
        # seen_image_ids = np.unique(mappings.values[0])
        seen_image_ids = lexunique(mappings.images)
        images = images[seen_image_ids]

        # Update image indices to the new images length. This is
        # important for preserving the mappings and for multimodal data
        # batching mechanisms.
        # mappings.values[0] = np.digitize(mappings.values[0], seen_image_ids) - 1
        mappings.images = torch.bucketize(mappings.images, seen_image_ids)

        return data, images, mappings

    def __call__(self, data, images, mappings):
        """
        Populate data sample in place with image attributes in mappings,
        based on the self.key point identifiers.
        """
        if isinstance(data, list):
            if isinstance(images, list) and isinstance(mappings, list) and \
                    len(images) == len(data) and len(mappings) == len(data):
                out = [self._process(d, i, m)
                       for d, i, m in zip(data, images, mappings)]
            else:
                out = [self._process(d, images, mappings) for d in data]
            data, images, mappings = [list(x) for x in zip(*out)]

        else:
            data, images, mappings = self._process(data, images, mappings)

        return data, images, mappings

    def __repr__(self):
        return self.__class__.__name__


class TorchvisionTransform:
    """Torchvision-based transform on the images"""

    def __init__(self):
        raise NotImplementedError

    def __call__(self, data, images, mappings):
        images.x = self.transform(images.x)
        return data, images, mappings

    def __repr__(self):
        return self.transform.__repr__()


class ColorJitter(TorchvisionTransform):
    """Randomly change the brightness, contrast and saturation of an image."""

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.transform = T.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation,
            hue=hue)


class RandomHorizontalFlip(TorchvisionTransform):
    """Horizontally flip the given image randomly with a given probability."""

    def __init__(self, p=0.50):
        self.p = p
        self.transform = T.RandomHorizontalFlip(p=p)


class GaussianBlur(TorchvisionTransform):
    """Blurs image with randomly chosen Gaussian blur."""

    def __init__(self, kernel_size=10, sigma=(0.1, 2.0)):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.transform = T.GaussianBlur(kernel_size, sigma=sigma)


# TODO : add invertible transforms from https://github.com/gregunz/invertransforms
#  or modify the mappings when applying the geometric transforms.
#  WARNING : if the image undergoes geometric transform, this may cause
#  problems when doing image wrapping or in EquiConv. IDEA : spherical
#  image rotation for augmentation

# TODO: consider impact of padding on mappings. Should we pad only once
#  as an image transform operation or should we pad in the convolution
#  blocks ? It seems cleaner to do it in the convolutions, rather doing
#  only 1 big padding at the beginning (that would require knowing how
#  many conv layers the image will go through). So, how and where do we
#  edit the mappings based on the paddings ?
