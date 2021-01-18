import numpy as np
import torch
import torch_scatter
from torch_geometric.data import Data
from torch_points3d.core.data_transform import SphereSampling
from torch_points3d.datasets.multimodal.image import ImageData, ImageMapping, \
    ImageDataList
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


# TODO: edit all transforms to have mappings inside the ImageData and handle ImageDataList
# TODO: edit S3DIS accordingly

class NonStaticMask:
    """
    Transform-like structure. Find the mask of non-identical pixels across
    a list of images.
    """

    def __init__(self, ref_size=(512, 256), proj_upscale=2, n_sample=5):
        self.ref_size = tuple(ref_size)
        self.proj_upscale = proj_upscale
        self.n_sample = n_sample

    def _process(self, images):
        # Edit ImageData 'ref_size' and 'proj_upscale' attributes. This
        # will make the projection mask size accessible as 'proj_size'
        images.ref_size = self.ref_size
        images.proj_upscale = self.proj_upscale

        # Compute the mask of identical pixels across a range of
        # 'n_sample' sampled images
        n_sample = min(self.n_sample, images.num_images)
        if n_sample < 2:
            mask = torch.ones(images.proj_size, dtype=torch.bool)
        else:
            # Initialize the mask to all False (ie all identical)
            mask = torch.zeros(images.proj_size, dtype=torch.bool)

            # Iteratively update the mask w.r.t. a reference image
            idx = torch.multinomial(
                torch.arange(images.num_images, dtype=torch.float), n_sample)

            # Read individual RGB images and squeeze them shape 3xHxW
            img_1 = images.read_images(idx=idx[0], size=images.proj_size).squeeze()

            for i in idx[1:]:
                img_2 = images.read_images(idx=i, size=images.proj_size).squeeze()

                # Update mask where new pixel changes are detected
                mask_diff = (img_1 != img_2).all(dim=0).t()
                idx_diff_new = torch.where(torch.logical_and(mask_diff, ~mask))
                mask[idx_diff_new] = 1

        # Save the mask in the ImageData 'mask' attribute
        images.mask = mask

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


class MapImages:
    """
    Transform-like structure. Computes the mappings between individual
    3D points and image pixels. Point mappings are identified based on
    the self.key point identifiers.
    """

    def __init__(self, ref_size=(512, 256), proj_upscale=2, voxel=0.1, r_max=30,
                 r_min=0.5, growth_k=0.2, growth_r=10, empty=0, no_id=-1,
                 key='point_index'):

        self.key = key
        self.empty = empty
        self.no_id = no_id
        self.ref_size = ref_size
        self.proj_upscale = proj_upscale
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

        # Edit ImageData projection attributes attributes. This
        # will make the projection mask size accessible as 'proj_size'
        images.ref_size = self.ref_size
        images.proj_upscale = self.proj_upscale
        images.voxel = self.voxel
        images.r_max = self.r_max
        images.r_min = self.r_min
        images.growth_k = self.growth_k
        images.growth_r = self.growth_r

        if images.mask is not None:
            assert images.mask.shape == images.proj_size

        # Initialize the mapping arrays
        image_ids = []
        point_ids = []
        pixels = []

        from time import time
        t_sphere_sampling = 0
        t_projection = 0
        t_init_torch_pixels = 0
        t_coord_pixels = 0
        t_unique_pixels = 0
        t_stack_pixels = 0
        t_append = 0

        # Project each image and gather the point-pixel mappings
        for i_image, image in tq(enumerate(images)):
            # Subsample the surrounding point cloud
            start = time()
            sampler = SphereSampling(image.r_max, image.pos, align_origin=False)
            data_sample = sampler(data)
            t_sphere_sampling += time() - start

            # Projection to build the index map
            start = time()
            id_map, _ = compute_index_map(
                (data_sample.pos - image.pos.squeeze()).numpy(),
                getattr(data_sample, self.key).numpy(),
                np.array(image.opk.squeeze()),
                img_mask=image.mask.numpy() if image.mask is not None else None,
                proj_size=image.proj_size,
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
            point_ids_ = id_map[(pix_x_, pix_y_)]

            # Skip image if no mapping was found
            if point_ids_.shape[0] == 0:
                continue
            t_init_torch_pixels += time() - start

            # Convert to ImageData coordinate system with proper
            # resampling and cropping
            # TODO: add circular padding here if need be
            start = time()
            pix_x_ = (pix_x_ // image.proj_upscale).long()
            pix_y_ = (pix_y_ // image.proj_upscale).long()
            pix_x_ = pix_x_ - image.crop_offsets.squeeze()[0]
            pix_y_ = pix_y_ - image.crop_offsets.squeeze()[1]
            cropped_in_idx = torch.where(
                (pix_x_ >= 0)
                & (pix_y_ >= 0)
                & (pix_x_ < image.crop_size[0])
                & (pix_y_ < image.crop_size[1]))
            pix_x_ = pix_x_[cropped_in_idx]
            pix_x_ = pix_x_[cropped_in_idx]
            point_ids_ = point_ids_[cropped_in_idx]
            pix_x_ = (pix_x_ // image.downscale).long()
            pix_y_ = (pix_y_ // image.downscale).long()
            t_coord_pixels += time() - start

            # Remove duplicate id-xy in low resolution
            # Sort by point id
            start = time()
            point_ids_, pix_x_, pix_y_ = lexunique(point_ids_, pix_x_, pix_y_)
            t_unique_pixels += time() - start

            # Cast pixel coordinates to a dtype minimizing memory use
            start = time()
            pixels_ = torch.stack((pix_x_, pix_y_), dim=1).type(
                image.pixel_dtype)
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
        print(f"        t_coord_pixels: {t_coord_pixels:0.3f}")
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
        start = time()
        seen_image_ids = lexunique(image_ids)
        images = images[seen_image_ids]
        image_ids = torch.bucketize(image_ids, seen_image_ids)
        print(f"        t_index_image_data: {time() - start:0.3f}")

        # Concatenate mappings data
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


class DropUninformativeImages:
    """
    Transform to drop images and corresponding mappings when mappings
    account for less than a given ratio of the image area.
    """

    def __init__(self, ratio=0.03):
        self.ratio = ratio

    def _process(self, images):
        # Threshold below which the number of pixels in the mappings
        # is deemed insufficient
        threshold = images.img_size[0] * images.img_size[1] * self.ratio

        # Count the number of pixel mappings for each image
        image_ids = torch.repeat_interleave(
            images.mappings.images,
            images.mappings.values[1].pointers[1:]
            - images.mappings.values[1].pointers[:-1])
        image_counts = torch_scatter.scatter_add(
            torch.ones(image_ids.shape[0]), image_ids, dim=0)

        # Select the images and mappings meeting the threshold
        return images[torch.where(image_counts > threshold)]

    def __call__(self, data, images, mappings):
        if isinstance(images, list):
            images = [self._process(img) for img in images]
        else:
            images = self._process(images)
        return data, images, mappings

    def __repr__(self):
        return self.__class__.__name__


class CropImageGroups:
    """
    Transform to crop images and mappings in a greedy fashion, so as to
    minimize the size of the images while preserving all the mappings and
    padding constraints. This is typically useful for optimizing the size of
    the images to embed with respect to the available mappings.

    The images are distributed to a set of cropping sizes, based on their
    mappings and the padding. Images with the same cropping size are batched
    together.

    Returns a list of ImageData of fixed cropping sizes with their respective
    mappings.
    """

    def __init__(self, padding=0, min_size=64):
        assert padding >= 0, \
            f"Expected a positive scalar but got {padding} instead."
        assert ((min_size & (min_size - 1)) == 0) & (min_size != 0), \
            f"Expected a power of two but got {min_size} instead."
        self.padding = padding
        self.min_size = min_size

    def _process(self, images: ImageData) -> ImageDataList:
        # Compute the bounding boxes for each image
        boxes = images.mappings.bounding_boxes

        # Add padding to the boxes
        boxes[:, 0] = torch.clamp(boxes[:, 0] - self.padding, 0)
        boxes[:, 2] = torch.clamp(boxes[:, 2] - self.padding, 0)
        boxes[:, 1] = torch.clamp(boxes[:, 1] + self.padding, 0,
                                  images.img_size[0])
        boxes[:, 3] = torch.clamp(boxes[:, 3] + self.padding, 0,
                                  images.img_size[1])
        boxes_width = boxes[:, 2] - boxes[:, 0]
        boxes_height = boxes[:, 3] - boxes[:, 1]

        # Compute the family of possible crop sizes and assign each
        # image to the relevant one.
        # The first size is (min_size, min_size). The other ones follow:
        # (min_size * 2a, min_size * 2^b)), with a = ^^ or a = b+1
        # The last crop size is the full img_size.
        crop_families = {}
        size = (self.min_size, self.min_size)
        i_crop = 0
        image_ids = torch.arange(images.num_images)
        while all(a <= b for a, b in zip(size, images.img_size)):
            # Safety measure to make sure all images are used
            if size == images.img_size:
                crop_families[size] = image_ids
                break

            # Search among the remaining images those that would fit in
            # the crop size
            valid_ids = boxes_width[image_ids] <= size[0] \
                        & boxes_height[image_ids] <= size[1]
            crop_families[size] = image_ids[valid_ids]

            # Discard selected image ids
            image_ids = image_ids[~valid_ids]

            # Compute the next the size
            size = (size[0] * 2 ** ((i_crop + 1) % 2), size[1] * 2 ** (i_crop % 2))
            i_crop += 1

        # Make sure the last crop size is the full image
        if images.img_size not in crop_families.keys():
            crop_families[images.img_size] = image_ids

        # Index and crop the images and mappings
        for size, idx in crop_families:
            # Compute the crop offset for each image
            off_x = torch.clamp(
                (boxes[idx, 0] + (boxes_width[idx] - size[0] / 2.)).long(),
                0, images.img_size[0] - size[0])
            off_y = torch.clamp(
                (boxes[idx, 2] + (boxes_height[idx] - size[1] / 2.)).long(),
                0, images.img_size[1] - size[1])
            offsets = torch.stack((off_x, off_y), dim=1).long()

            # Index images and mappings and update their cropping
            crop_families[size] = images[idx].update_cropping(size, offsets)

        # Create a holder for the ImageData of each crop size
        return ImageDataList(crop_families.values())

    def __call__(self, data, images, mappings):
        if isinstance(images, list):
            images = [self._process(img) for img in images]
        else:
            images = self._process(images)
        return data, images, mappings

    def __repr__(self):
        return self.__class__.__name__


class CropFromMask:
    """Transform to crop top and bottom from images and mappings based on mask."""
    # TODO: CropFromMask
    pass


class PadMappings:
    """Transform to update the mappings to account for image padding."""
    # TODO: PadMappings
    #  https://distill.pub/2019/computing-receptive-fields/
    #  https://github.com/google-research/receptive_field
    #  https://github.com/Fangyh09/pytorch-receptive-field
    #  https://github.com/rogertrullo/Receptive-Field-in-Pytorch/blob/master/compute_RF.py
    #  https://fomoro.com/research/article/receptive-field-calculator#3,1,1,SAME;3,1,1,SAME;2,2,1,SAME;3,1,1,SAME;3,1,1,SAME;2,2,1,SAME;3,1,1,SAME;3,1,1,SAME
    pass


class YFeature:
    # TODO: YFeature
    pass


class XFeature:
    # TODO: YFeature
    pass


# TODO: integrate this in MMData indexing ?
class ImageMappingFromPointId:
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
        seen_image_ids = lexunique(mappings.images)
        images = images[seen_image_ids]

        # Update image indices to the new images length. This is
        # important for preserving the mappings and for multimodal data
        # batching mechanisms.
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
        # TODO: use torch.flip(x, [2]) instead, to manually control the
        #  flip's randomness and edit the mappings accordingly


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
