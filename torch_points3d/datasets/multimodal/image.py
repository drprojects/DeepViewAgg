import os.path as osp
import numpy as np
from PIL import Image



class ImageData(object):

    # Class attributes
    mask = None
    img_size = (2048, 1024)
    map_size_high = (2048, 1024)
    __map_size_low__ = (512, 256)
    map_dtype = np.uint8
    crop_top = 0
    crop_bottom = 0
    voxel = 0.1
    r_max = 30
    r_min = 0.5
    growth_k = 0.2
    growth_r = 10


    def __init__(self, id, path, pos, opk):
        self.id = id
        self.pos = pos
        self.opk = opk
        self.path = path
        self.name = osp.splitext(osp.basename(path))[0]
#         self.image = self.read_image(self.path, self.size)

    
    def __repr__(self): 
        return self.__class__.__name__

    
    @staticmethod
    def read_image(path, size):
        return np.array(Image.open(path).convert('RGB').resize(size, Image.LANCZOS))


    @staticmethod
    def coarsen_coordinates(x_high, res_high, res_low, dtype):
        """Convert pixel coordinates from high to low resolution."""
        ratio = res_low / res_high
        return np.floor(x_high * ratio).astype(dtype)


    @property
    def map_size_low(self):
        return ImageData.__map_size_low__ 


    @map_size_low.setter
    def map_size_low(map_size_low):
        ImageData.__map_size_low__ = map_size_low

        # Update the optimal xy mapping dtype allowed by the resolution
        for dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
            if np.iinfo(dtype).max >= max(ImageData.map_size_low[0], ImageData.map_size_low[1]):
                break
        ImageData.map_dtype = dtype


    @staticmethod
    def non_static_pixel_mask(img_list, mask_size, n_sample):
        """
        Find the mask of identical pixels accross a list of images.
        """
        for image in img_list:
            assert isinstance(image, ImageData)

        # Iteratively update the mask w.r.t. a reference image
        mask = np.ones(mask_size, dtype='bool')
        img_1 = ImageData.read_image(image.path, mask_size)

        n_sample = min(n_sample, len(img_list)-1)
        img_list_sample = np.random.choice(img_list[1:], size=n_sample, replace=False)

        for img_path in img_list_sample:
            img_2 = ImageData.read_image(image.path, mask_size)
            mask_equal = np.all(img_1 == img_2, axis=2).transpose()
            mask[np.logical_and(mask, mask_equal)] = 0

        return mask