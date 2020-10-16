import numpy as np
from torch_geometric.data import Data
from torch_points3d.core.data_transform import SphereSampling
from torch_points3d.datasets.multimodal.image import ImageData
from torch_points3d.datasets.multimodal.forward_star import ForwardStar
from .projection import compute_index_map



"""
Image-based transforms for multimodal data processing. Inspired by 
torch_points3d and torch_geometric transforms on data, with a signature 
allowing for multimodal transform composition : __call(data, images, mappings)__
"""



class NonStaticImageMask(object):
    """
    Transform-like structure. Find the mask of identical pixels accross a list
    of images.
    """
    def __init__(self, mask_size=(2048, 1024), n_sample=5):
        self.mask_size = mask_size
        self.n_sample = n_sample


    def __call__(self, data, images, mappings=None):
        """
        Compute the projection of data points into images and return the input 
        data augmented with attributes mapping points to pixels in provided 
        images.

        Expects Data (or anything), List(ImageData) or List(List(ImageData)), 
        ForwardStar mapping (or anything).

        Returns the same input. The mask is saved in class attributes of
        ImageData, to be used for any subsequent image processing.
        """
        assert isinstance(images, list)
        if isinstance(images[0], list):
            flat_images_list = [im for im_sublist in images for im in im_sublist]
        else:
            flat_images_list = images

        mask = ImageData.non_static_pixel_mask(flat_images_list, self.mask_size, self.n_sample)
        ImageData.mask = mask

        return data, images, mappings


    def __repr__(self):
        return self.__class__.__name__
    
#-------------------------------------------------------------------------------

class PointImagePixelMapping(object):
    """
    Transform-like structure. Computes the mappings between individual 3D points
    and image pixels. Point mappings are identified based on the self.key point
    identifiers.
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
            key='processed_id'
        ):

        self.key = key
        self.empty = empty
        self.no_id = no_id

        # Store the projection parameters in the ImageData class attributes.
        ImageData.map_size_high = tuple(map_size_high)
        ImageData.map_size_low = tuple(map_size_low)
        ImageData.crop_top = crop_top
        ImageData.crop_bottom = crop_bottom
        ImageData.voxel = voxel
        ImageData.r_max = r_max
        ImageData.r_min = r_min
        ImageData.growth_k = growth_k
        ImageData.growth_r = growth_r

        if ImageData.mask is not None:
            assert ImageData.mask.shape == ImageData.map_size_high

  
    def _process(self, data, images):

        assert hasattr(data, self.key)

        # Initialize the
        image_ids = []
        point_ids = []
        pixels = []

        # Project each image and gather the point-pixel mappings
        for i_image, image in enumerate(images):

            print(f"Image {i_image} : '{image.name}'")

            # Subsample the surrounding point cloud
            sampler = SphereSampling(image.r_max, image.pos, align_origin=False)
            data_sample = sampler(data.clone())

            # Projection index
            id_map, _ = compute_index_map(
                data_sample.pos.numpy() - image.pos,
                getattr(data_sample, self.key).numpy(),
                image.opk,
                img_mask=image.mask,
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

            # Convert the id_map to id-xy coordinate soup
            # First column holds the point indices, subsequent columns hold the 
            # pixel coordinates. We use this heterogeneous soup to search for 
            # duplicate rows after resolution coarsening.
            # NB : no_id pixels are ignored
            active_pixels = np.where(id_map != self.no_id)
            point_ids_pixel_soup = id_map[active_pixels]
            point_ids_pixel_soup = np.column_stack((point_ids_pixel_soup,
                np.stack(active_pixels).transpose()))

            # Convert to lower resolution coordinates
            # NB: we assume the resolution ratio is the same for both 
            # dimensions 
            point_ids_pixel_soup[:, 1:] = ImageData.coarsen_coordinates(point_ids_pixel_soup[:, 1:],
                ImageData.map_size_high[0], ImageData.map_size_low[0], ImageData.map_dtype)

            # Remove duplicate id-xy in low resolution
            # Sort by point id
            point_ids_pixel_soup = np.unique(point_ids_pixel_soup, axis=0)  # bottleneck here ! Custom unique-sort with numba ?

            # Cast pixel coordinates to a dtype minimizing memory use
            point_ids_ = point_ids_pixel_soup[:, 0]
            pixel_ = point_ids_pixel_soup[:, 1:].astype(ImageData.map_dtype)
            del point_ids_pixel_soup

            # Gather per-image mappings in list structures, only to be
            # numpy-stacked once all images are processed
            image_ids.append(i_image)
            point_ids.append(point_ids_)
            pixels.append(pixel_)
            del pixel_, point_ids_

        # Concatenate mappings
        image_ids = np.repeat(image_ids, [x.shape[0] for x in point_ids])
        point_ids = np.concatenate(point_ids)
        pixels = np.vstack(pixels)

        # Sort by point_ids first, image_ids second
        sorting = np.lexsort((image_ids, point_ids))
        image_ids = image_ids[sorting]
        point_ids = point_ids[sorting]
        pixels = pixels[sorting]
        del sorting

        # Convert to "nested Forward Star" format
        # Compute image jumps in the pixels array
        image_pixel_mappings = ForwardStar(image_ids, pixels, dense=True)
        
        # Update point_ids and image_ids by taking the last value of each jump
        image_ids = image_ids[image_pixel_mappings.jumps[1:] - 1]
        point_ids = point_ids[image_pixel_mappings.jumps[1:] - 1]

        # Compute point jumps in the image_ids array
        point_image_mappings = ForwardStar(point_ids, image_ids, image_pixel_mappings, dense=True)

        # Update point_ids by taking the last value of each jump
        point_ids = point_ids[point_image_mappings.jumps[1:] - 1]

        # Some points may have been seen by no image so we need to inject 
        # 0-sized jumps to account for these.
        # NB: we assume all relevant points are present in data.processed_id, 
        #     if a point with an id larger than max(data.processed_id) were to 
        #     exist, we would not be able to take it into account in the jumps.
        num_points = getattr(data, self.key).numpy().max() + 1
        point_image_mappings = point_image_mappings.reindex_groups(point_ids, num_groups=num_points)

        return point_image_mappings


    def __call__(self, data, images, mappings=None):
        """
        Compute the projection of data points into images and return the input 
        data augmented with attributes mapping points to pixels in provided 
        images.

        Expects a Data and a List(ImageData) or a List(Data) and a 
        List(List(ImageData)) of matching lengths.

        Returns the input data and the point-image-pixel mappings in a nested 
        ForwardStar format.
        """
        assert isinstance(images, list)

        if isinstance(data, list):
            assert len(data) == len(images), (f"List(Data) items and List(List(ImageData)) must ",
                "have the same lengths.")
            mappings = [self._process(d, i) for d, i in zip(data, images)]

        else:
            mappings = self._process(data, images)

        return data, images, mappings


    def __repr__(self):
        return self.__class__.__name__

#-------------------------------------------------------------------------------

class PointImagePixelMappingFromId(object):
    """
    Transform-like structure. Intended to be called on _datas and _images_datas.

    Populate the passed Data sample in-place with attributes extracted from the 
    input ForwardStar mappings, based on the self.key point identifiers.
    
    The indices in data are expected to be included in those in mappings. The 
    ForwardStar format implicitly holds values for all self.key in 
    [0, ..., len(mappings)].
    """
    def __init__(self, key='processed_id'):
        self.key = key


    def _process(self, data, mappings):
        assert hasattr(data, self.key)
        assert isinstance(mappings, ForwardStar)

        # Point indices to subselect point_jumps 
        indices = getattr(data, self.key)

        # Subselect mappings with ForwardStar indexing 
        data_mappings = mappings[indices]

        # Populate data with attribute names making use of torch_geometric.Data 
        # and torch_geometric.Batch special mechanisms for "*index*" attributes. 
        settatr(data, torch.from_array(data_mappings.jumps), 'point_jump_index')
        settatr(data, torch.from_array(data_mappings.values[0]), 'image_ids')
        settatr(data, torch.from_array(data_mappings.values[1].jumps), 'image_jump_index')
        settatr(data, torch.from_array(data_mappings.values[1].values[0]), 'pixels')

        return data


    def __call__(self, data, images, mappings):
        """
        Populate data sample in place with image attributes in mappings,
        based on the self.key point identifiers.
        """
        if isinstance(data, list):
            if isinstance(mappings, ForwardStar):
                data = [self._process(d, mappings) for d in data]
            else:
                assert len(data) == len(mappings)
                data = [self._process(d, m) for d, m in zip(data, mappings)]
        else:
            assert isinstance(mappings, ForwardStar)
            data = self._process(data, mappings)
        return data, images, mappings


    def __repr__(self):
        return self.__class__.__name__

