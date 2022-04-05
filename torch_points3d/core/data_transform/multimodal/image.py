import os.path as osp
import torch
import numpy as np
import torch_scatter
from torch_geometric.data import Data
from torch_points3d.core.data_transform import SphereSampling, \
    CylinderSampling, GridSampling3D, SaveOriginalPosId
from torch_points3d.core.spatial_ops.neighbour_finder import \
    FAISSGPUKNNNeighbourFinder
from torch_points3d.utils.multimodal import MAPPING_KEY
from torch_points3d.core.multimodal.image import SameSettingImageData, \
    ImageMapping, ImageData
from torch_points3d.utils.multimodal import lexunique, lexargunique
import torch_points3d.core.multimodal.visibility as visibility_module
import torchvision.transforms as T
from tqdm.auto import tqdm as tq
from typing import TypeVar, Union
from pykeops.torch import LazyTensor
from collections.abc import Iterable

"""
Image-based transforms for multimodal data processing. Inspired by 
torch_points3d and torch_geometric transforms on data, with a signature 
allowing for multimodal transform composition: 
__call(Data, ImageData)__
"""


class ImageTransform:
    """Transforms on `Data`, `ImageData` / `SameSettingImageData` and 
    associated `ImageMapping`.
    """

    _PROCESS_IMAGE_DATA = False

    def _process(self, data: Data,
                 images: Union[ImageData, SameSettingImageData]):
        raise NotImplementedError

    def __call__(self, data, images):
        if isinstance(data, list):
            assert isinstance(images, list) and len(data) == len(images), \
                f"List(Data) items and List(SameSettingImageData) must " \
                f"have the same lengths."
            out = [self.__call__(da, im) for da, im in zip(data, images)]
            data_out, images_out = [list(x) for x in zip(*out)]
        elif isinstance(images, ImageData) and not self._PROCESS_IMAGE_DATA:
            out = [self.__call__(data.clone(), im) for im in images]
            images_out = ImageData([im for _, im in out])
            data_out = out[0][0] if len(out) > 0 else data
        else:
            if isinstance(images, SameSettingImageData) \
                    and self._PROCESS_IMAGE_DATA:
                images = ImageData([images])
            # data_out, images_out = self._process(data.clone(), images.clone())
            data_out, images_out = self._process(data, images)
        return data_out, images_out

    def __repr__(self):
        attr_repr = ', '.join([f'{k}={v}' for k, v in self.__dict__.items()])
        return f'{self.__class__.__name__}({attr_repr})'


class ToImageData(ImageTransform):
    """Transform to convert `SameSettingImageData` into `ToImageData`.
    """
    def _process(self, data: Data, images: SameSettingImageData):
        return data, ImageData([images])


class LoadImages(ImageTransform):
    """Transform to load images from disk to the `SameSettingImageData`.

    `SameSettingImageData` internal state is updated if resizing and
    cropping parameters are passed. Images are loaded with respect to
    the `SameSettingImageData` resizing and cropping internal state.
    """

    def __init__(self, ref_size=None, crop_size=None, crop_offsets=None,
                 downscale=None, show_progress=False):
        self.ref_size = ref_size
        self.crop_size = crop_size
        self.crop_offsets = crop_offsets
        self.downscale = downscale
        self.show_progress = show_progress

    def _process(self, data: Data, images: SameSettingImageData):
        # Edit `SameSettingImageData` internal state attributes.
        if self.ref_size is not None:
            images.ref_size = self.ref_size
        if self.crop_size is not None:
            images.crop_size = self.crop_size
        if self.crop_offsets is not None:
            images.crop_offsets = self.crop_offsets
        if self.downscale is not None:
            images.downscale = self.downscale

        # Load images wrt `SameSettingImageData` internal state
        if self.show_progress:
            print("    LoadImages...")
        images.load(show_progress=self.show_progress)

        return data, images


class NonStaticMask(ImageTransform):
    """Transform-like structure. Find the mask of non-identical pixels
    across a list of images.

    Compute the projection of data points into images and return the
    input data augmented with attributes mapping points to pixels in
    provided images.

    Returns the same input. The mask is saved in `SameSettingImageData`
    attributes, to be used for any subsequent image processing.
    """

    def __init__(self, ref_size=None, proj_upscale=None, n_sample=5):
        self.ref_size = tuple(ref_size)
        self.proj_upscale = proj_upscale
        self.n_sample = n_sample

    def _process(self, data: Data, images: SameSettingImageData):
        # Edit `SameSettingImageData` internal state attributes
        if self.ref_size is not None:
            images.ref_size = self.ref_size
        if self.proj_upscale is not None:
            images.proj_upscale = self.proj_upscale

        # Compute the mask of identical pixels across a range of
        # 'n_sample' sampled images
        n_sample = min(self.n_sample, images.num_views)
        if n_sample < 2:
            mask = torch.ones(images.proj_size, dtype=torch.bool)
        else:
            # Initialize the mask to all False (ie all identical)
            mask = torch.zeros(images.proj_size, dtype=torch.bool)

            # Iteratively update the mask w.r.t. a reference image
            idx = torch.multinomial(
                torch.arange(images.num_views, dtype=torch.float), n_sample)

            # Read individual RGB images and squeeze them shape 3xHxW
            img_1 = images.read_images(
                idx=idx[0], size=images.proj_size).squeeze()

            for i in idx[1:]:
                img_2 = images.read_images(
                    idx=i, size=images.proj_size).squeeze()

                # Update mask where new pixel changes are detected
                mask_diff = (img_1 != img_2).all(dim=0).t()
                idx_diff_new = torch.where(torch.logical_and(mask_diff, ~mask))
                mask[idx_diff_new] = 1

        # Save the mask in the `SameSettingImageData` 'mask' attribute
        images.mask = mask

        return data, images


class MapImages(ImageTransform):
    """Transform-like structure. Computes the mappings between
    individual 3D points and image pixels. Point mappings are identified
    based on the `self.key` point identifiers.

    Compute the projection of data points into images and return the
    input data augmented with attributes mapping points to pixels in
    provided images.

    Returns the input data and `SameSettingImageData` augmented with the
    point-image-pixel `ImageMapping`.
    """

    def __init__(
            self, method='SplattingVisibility', proj_upscale=None,
            ref_size=None, use_cuda=False, verbose=False, cylinder=False,
            **kwargs):
        self.key = MAPPING_KEY
        self.verbose = verbose
        self.cylinder = cylinder

        # Image internal state parameters
        self.ref_size = ref_size
        self.proj_upscale = proj_upscale

        # Visibility model parameters
        self.method = method
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.kwargs = kwargs

    def _process(self, data: Data, images: SameSettingImageData):
        assert hasattr(data, self.key)
        assert isinstance(images, SameSettingImageData)
        assert images.num_views >= 1, \
            "At least one image must be provided."

        # Initialize the input-output device and the device on which
        # heavy computation should be performed
        in_device = images.device
        device = 'cuda' if self.use_cuda else in_device

        # Edit `SameSettingImageData` internal state attributes
        if self.ref_size is not None:
            images.ref_size = self.ref_size
        if self.proj_upscale is not None:
            images.proj_upscale = self.proj_upscale

        # Control the size of any already-existing mask
        if images.mask is not None:
            assert images.mask.shape == images.proj_size

        # Instantiate the visibility model
        visi_cls = getattr(visibility_module, self.method)
        visi_model = visi_cls(img_size=images.proj_size, **self.kwargs)

        # Initialize the mapping arrays
        image_ids = []
        point_ids = []
        features = []
        pixels = []

        from time import time
        t_sphere_sampling = 0
        t_visibility = 0
        t_coord_pixels = 0
        t_unique_pixels = 0
        t_stack_pixels = 0
        t_append = 0

        # Project each image and gather the point-pixel mappings (on
        # CPU or GPU)
        if self.verbose:
            print("    MapImages...")
            enumerator = tq(enumerate(images))
        else:
            enumerator = enumerate(images)
        for i_image, image in enumerator:
            # Subsample the surrounding point cloud
            torch.cuda.synchronize()
            start = time()
            cls = CylinderSampling if self.cylinder else SphereSampling
            center = image.pos.squeeze()[:2] if self.cylinder else image.pos
            sampler = cls(visi_model.r_max, center, align_origin=False)
            data_sample = sampler(data)
            torch.cuda.synchronize()
            t_sphere_sampling += time() - start

            # Prepare the visibility model input parameters
            linearity = getattr(data_sample, 'linearity', None)
            planarity = getattr(data_sample, 'planarity', None)
            scattering = getattr(data_sample, 'scattering', None)
            normals = getattr(data_sample, 'norm', None)
            linearity = linearity.to(device) if linearity is not None else None
            planarity = planarity.to(device) if planarity is not None else None
            scattering = scattering.to(device) if scattering is not None else None
            normals = normals.to(device) if normals is not None else None

            # TEMPORARY - read depth map from file for S3DIS images
            # TODO: better handle depth map files if DepthMapBasedVisibility is
            #  needed for other datasets than S3DIS
            depth_map_path = osp.join(
                osp.dirname(osp.dirname(image.path[0])), 'depth',
                osp.basename(image.path[0]).replace('_rgb.png', '_depth.png'))

            # Compute the visibility of points wrt camera pose. This
            # provides us with indices of visible points along with
            # corresponding pixel coordinates, depth and mapping
            # features.
            # (on CPU or GPU)
            torch.cuda.synchronize()
            start = time()
            out_vm = visi_model(
                data_sample.pos.float().to(device),
                image.pos.squeeze().float().to(device),
                img_opk=image.opk.squeeze().float().to(device) if image.has_opk else None,
                img_intrinsic_pinhole=image.intrinsic_pinhole.squeeze().float().to(device) if image.is_pinhole else None,
                img_intrinsic_fisheye=image.intrinsic_fisheye.squeeze().float().to(device) if image.is_fisheye else None,
                img_extrinsic=image.extrinsic.squeeze().float().to(device) if image.has_extrinsic else None,
                img_mask=image.mask.to(device) if image.mask is not None else None,
                linearity=linearity.to(device) if linearity is not None else None,
                planarity=planarity.to(device) if planarity is not None else None,
                scattering=scattering.to(device) if scattering is not None else None,
                normals=normals.to(device) if normals is not None else None,
                depth_map_path=depth_map_path)
            del linearity, planarity, scattering, normals

            # Skip image if no mapping was found
            if out_vm['idx'].shape[0] == 0:
                continue

            # Recover point indices, pixel coordinates and features
            # (on CPU or GPU)
            point_ids_ = data_sample[self.key].to(device)[out_vm['idx']]
            pix_x_ = out_vm['x'].long()
            pix_y_ = out_vm['y'].long()
            features_ = out_vm['features'].float()
            del out_vm, data_sample
            torch.cuda.synchronize()
            t_visibility += time() - start

            # Convert to `SameSettingImageData` coordinate system with
            # corresponding cropping and resizing
            # (on CPU or GPU)
            # TODO: add circular padding here if need be
            start = time()
            pix_x_ = pix_x_ // image.proj_upscale
            pix_y_ = pix_y_ // image.proj_upscale
            pix_x_ = pix_x_ - image.crop_offsets.squeeze()[0]
            pix_y_ = pix_y_ - image.crop_offsets.squeeze()[1]
            in_crop = torch.where(
                (pix_x_ >= 0) & (pix_y_ >= 0) & (pix_x_ < image.crop_size[0])
                & (pix_y_ < image.crop_size[1]))
            pix_x_ = pix_x_[in_crop]
            pix_y_ = pix_y_[in_crop]
            point_ids_ = point_ids_[in_crop]
            features_ = features_[in_crop]
            pix_x_ = (pix_x_ // image.downscale).long()
            pix_y_ = (pix_y_ // image.downscale).long()
            del in_crop
            torch.cuda.synchronize()
            t_coord_pixels += time() - start

            # Remove duplicate id-xy after image resizing
            # Sort by point id
            # (on CPU or GPU)
            start = time()
            unique_idx = lexargunique(point_ids_, pix_x_, pix_y_, use_cuda=True)
            pix_x_ = pix_x_[unique_idx]
            pix_y_ = pix_y_[unique_idx]
            point_ids_ = point_ids_[unique_idx]
            features_ = features_[unique_idx]
            del unique_idx
            torch.cuda.synchronize()
            t_unique_pixels += time() - start

            # Cast pixel coordinates to a dtype minimizing memory use
            # (on CPU or GPU)
            start = time()
            pixels_ = torch.stack((pix_x_, pix_y_), dim=1).type(
                image.pixel_dtype)
            del pix_x_, pix_y_
            torch.cuda.synchronize()
            t_stack_pixels += time() - start

            # Gather per-image mappings in list structures, only to be
            # torch-concatenated once all images are processed
            # (on CPU)
            start = time()
            image_ids.append(i_image)
            point_ids.append(point_ids_.cpu())
            features.append(features_.cpu())
            pixels.append(pixels_.cpu())
            del pixels_, features_, point_ids_
            torch.cuda.synchronize()
            t_append += time() - start

        if self.verbose:
            print(f"    Cumulated times")
            print(f"        t_sphere_sampling: {t_sphere_sampling:0.3f}")
            print(f"        t_visibility: {t_visibility:0.3f}")
            print(f"        t_coord_pixels: {t_coord_pixels:0.3f}")
            print(f"        t_unique_pixels: {t_unique_pixels:0.3f}")
            print(f"        t_stack_pixels: {t_stack_pixels:0.3f}")
            print(f"        t_append: {t_append:0.3f}")

        # Drop the KD-tree attribute saved in data by the SphereSampling
        delattr(data, SphereSampling.KDTREE_KEY)

        # Raise error if no point-image-pixel mapping was found
        # (on CPU)
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
        # We want all images present in the mappings and in
        # `SameSettingImageData` to have been seen. If an image has not
        # been seen, we remove it here.
        # NB: The reindexing here relies on the fact that `unique`
        #  values are expected to be returned sorted.
        # (on CPU or GPU)
        torch.cuda.synchronize()
        start = time()
        seen_image_ids = lexunique(image_ids, use_cuda=self.use_cuda)
        images = images[seen_image_ids.to(in_device)]
        image_ids = torch.bucketize(image_ids, seen_image_ids)
        del seen_image_ids
        if self.verbose:
            torch.cuda.synchronize()
            print(f"        t_index_image_data: {time() - start:0.3f}")

        # Concatenate mappings data (on CPU) and move them to required
        # device (on CPU or GPU)
        start = time()
        image_ids = image_ids.repeat_interleave(
            torch.LongTensor([x.shape[0] for x in point_ids])).to(device)
        point_ids = torch.cat(point_ids).to(device)
        pixels = torch.cat(pixels).to(device)
        features = torch.cat(features).to(device)
        if self.verbose:
            print(f"        t_concat_dense_mappings_data: {time() - start:0.3f}")

        # Compute the global `ImageMapping`
        # (on CPU or GPU)
        torch.cuda.synchronize()
        start = time()
        mappings = ImageMapping.from_dense(
            point_ids, image_ids, pixels, features,
            num_points=getattr(data, self.key).numpy().max() + 1)
        del point_ids, image_ids, pixels, features
        if self.verbose:
            torch.cuda.synchronize()
            print(f"        t_ImageMapping_init: {time() - start:0.3f}\n\n")

        # Save the mappings and visibility model in the
        # SameSettingImageData
        images.mappings = mappings.to(in_device)
        images.visibility = visi_model

        return data, images


class NeighborhoodBasedMappingFeatures(ImageTransform):
    """Transform-like structure. Intended to be called on data and
    images_data.

    Populate the mappings with neighborhood-based mapping features:
        - density: estimated with the volume of the K-NN sphere
        - occlusion: estimated as the ratio of k-NN seen by the image at
        hand

    The indices in data are expected to be included in those in
    mappings. The CSRData format implicitly holds values for all
    self.key in [0, ..., len(mappings)].

    Parameters
    ----------
    k: int or List
        Controls the number of neighbors on which to compute the
        features. If a List is passed, the features will be computed for
        each value of k.
    voxel: float, optional
        Voxel resolution of the cloud. If provided and `density=True`,
        will be used to normalize the density. Because the surface
        density is approximated based on manifoldness assumptions, no
        guarantees can be claimed for normalized densities to be in
        [0, 1]. However, this scaling ensures the densities live in a
        "reasonable-enough" range to be subsequently fed to a neural
        network.
    density: bool, optional
        Whether the local densities should be computed.
    occlusion: bool, optional
        Whether the local occlusions should be computed.
    use_cuda: bool, optional
        If True, the computation will be carried on CUDA.
    """

    def __init__(
            self, k=20, voxel=None, density=True, occlusion=True,
            use_cuda=False, use_faiss=True, ncells=None, nprobes=10,
            verbose=False):
        self.k_list = sorted(k) if isinstance(k, list) else [k]
        self.voxel = voxel if voxel is not None else 1
        self.compute_density = density
        self.compute_occlusion = occlusion
        self.use_faiss = use_faiss and torch.cuda.is_available()
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.ncells = ncells
        self.nprobes = nprobes
        self.verbose = verbose
        assert density or occlusion, \
            "At least one of `density` or `occlusion` must be True."

    def _process(self, data: Data, images):
        if self.verbose:
            print("    NeighborhoodBasedMappingFeatures...")

        assert isinstance(data, Data)
        assert images.mappings is not None

        # Initialize the input-output device and the device on which
        # heavy computation should be performed
        in_device = images.device
        device = 'cuda' if self.use_cuda or self.use_faiss else in_device

        # Recover 3D points positions
        xyz = data.pos.to(device)

        # K-NN search
        if self.use_faiss:
            # K-NN search with FAISS
            if self.verbose:
                print(f"        KNN search with FAISS...")
            nn_finder = FAISSGPUKNNNeighbourFinder(
                self.k_list[-1], ncells=self.ncells, nprobes=self.nprobes)
            neighbors = nn_finder(xyz, xyz, None, None)
        else:
            # K-NN search with KeOps. If the number of points is greater
            # than 16 millions, KeOps requires double precision.
            if self.verbose:
                print(f"        KNN search with KeOps...")
            if xyz.shape[0] > 1.6e7:
                xyz_query_keops = LazyTensor(xyz[:, None, :].double())
                xyz_search_keops = LazyTensor(xyz[None, :, :].double())
            else:
                xyz_query_keops = LazyTensor(xyz[:, None, :])
                xyz_search_keops = LazyTensor(xyz[None, :, :])
            d_keops = ((xyz_query_keops - xyz_search_keops) ** 2).sum(dim=2)
            neighbors = d_keops.argKmin(self.k_list[-1], dim=1)
            del xyz_query_keops, xyz_search_keops, d_keops

        # Density computation
        # TODO: density is sensitive to noise level in the dataset, may affect
        #  generalization to other datasets if sensors differ. Investigating the
        #  effect of local noise may help compute a better density heuristic.
        if self.compute_density:
            if self.verbose:
                print(f"        Density computation...")

            densities = []
            for k in self.k_list:
                # Compute the farthest distance in each neighborhood
                d2_max = ((xyz - xyz[neighbors[:, k - 1]])**2).sum(dim=1)

                # Estimate the surface density. Points are assumed to
                # lie on a 2D manifold in 3D, so we roughly estimate the
                # 2D density based on the disk of radius d_max.
                v_sphere = 3.1416 * d2_max
                voxel_density = 1 / self.voxel**2
                density = ((k + 1) / v_sphere) / voxel_density

                # Set potential Nan densities to 1
                density[torch.where(density.isnan())] = 1

                # Accumulate on CPU to save GPU memory
                densities.append(density.cpu().view(-1, 1))

            # Concatenate k-based densities column-wise on the CPU
            densities = torch.cat(densities, dim=1)

            # Expand to view-level features
            densities = densities.to(in_device).repeat_interleave(
                images.mappings.pointers[1:] - images.mappings.pointers[:-1], 0)

            # Append densities to the image mapping features
            if not images.mappings.has_features:
                images.mappings.features = densities
            else:
                images.mappings.features = torch.cat(
                    [images.mappings.features, densities], dim=1)

        # Occlusion computation
        # TODO: occlusion could benefit from being computed on a spherical
        #  neighborhood rather than KNN. However, the density cannot be
        #  computed on a spherical neighborhood (because the neighbors are
        #  resampled to match a target number). This would mean computing
        #  occlusion and density separately, with their dedicated transforms,
        #  but this would also mean computing the neighbors twice...
        if self.compute_occlusion:
            if self.verbose:
                print(f"        Occlusion computation...")

            # Expand to view-level
            n_points = data.num_nodes
            n_images = torch.max(images.mappings.images) + 1
            pointers = images.mappings.pointers.to(device)
            point_ids = torch.arange(n_points, device=device).repeat_interleave(
                pointers[1:] - pointers[:-1])
            image_ids = images.mappings.images.to(device)

            # Compute the (very large) dense boolean 2D tensor of views
            views = torch.zeros(
                (n_points, n_images), dtype=torch.bool, device=device)
            views[point_ids, image_ids] = True

            occlusions = []
            for k in self.k_list:
                # Accumulate the number of seen neighbors each view
                views_neigh_seen = torch.ones_like(image_ids, dtype=torch.float)
                for i in range(k):
                    # Expand i-th neighbors to view-level
                    views_neigh = neighbors[:, i].repeat_interleave(
                        pointers[1:] - pointers[:-1])
                    views_neigh_seen += views[(views_neigh, image_ids)]

                # Recover the occlusion ratio for each view while
                # accounting for the contribution of the point itself
                occlusion = views_neigh_seen / (k + 1)

                # Accumulate on CPU to save GPU memory
                occlusions.append(occlusion.cpu().view(-1, 1))

            # Concatenate k-based occlusions column-wise on CPU before
            # moving to in_device
            occlusions = torch.cat(occlusions, dim=1).to(in_device)

            # Append occlusions to the image mapping features
            if not images.mappings.has_features:
                images.mappings.features = occlusions
            else:
                images.mappings.features = torch.cat(
                    [images.mappings.features, occlusions], dim=1)

        return data, images


class SelectMappingFromPointId(ImageTransform):
    """Transform-like structure. Intended to be called on data and
    images_data.

    Populate the passed `Data` sample with attributes extracted from the
    input `CSRData` mappings, based on the self.key point identifiers.

    The indices in data are expected to be included in those in
    mappings. The `CSRData` format implicitly holds values for all
    self.key in [0, ..., len(mappings)].
    """

    def __init__(self):
        self.key = MAPPING_KEY

    def _process(self, data, images):
        assert isinstance(data, Data)
        assert hasattr(data, self.key)
        assert isinstance(images, SameSettingImageData)
        assert images.mappings is not None

        # Select mappings and images wrt point key indices
        images = images.select_points(data[self.key], mode='pick')

        # Update point indices to the new mappings length. This is
        # important to preserve the mappings and for multimodal data
        # batching mechanisms.
        data[self.key] = torch.arange(data.num_nodes, device=images.device)

        return data, images


class DropImagesOutsideDataBoundingBox(ImageTransform):
    """Drop images that are not within the bounding box of data."""
    def __init__(self, margin=0, ignore_z=False):
        self.margin = margin
        self.ignore_z = ignore_z

    def _process(self, data: Data, images: SameSettingImageData):
        # Find the images that are withing the bounding box, with margin
        b_min = data.pos.min(dim=0).values - self.margin / 2
        b_max = data.pos.max(dim=0).values + self.margin / 2
        mask = torch.logical_and(b_min < images.pos, images.pos < b_max)
        if self.ignore_z:
            mask = mask[:, 0] * mask[:, 1]
        else:
            mask = mask[:, 0] * mask[:, 1] * mask[:, 2]

        # Select images
        images = images[mask]

        return data, images


class GridSampleImages(ImageTransform):
    """Grid-sample a set of images based on their 3D positions. This
    can be used to reduce an image set where close-by images are
    redundant.
    """
    def __init__(self, size=0):
        self.size = size

    def _process(self, data: Data, images: SameSettingImageData):
        # Create a Data object holding the image positions
        im_data = Data(pos=images.pos.clone())
        im_data = SaveOriginalPosId(key='image_id')(im_data)
        im_data = GridSampling3D(self.size, mode='last')(im_data)

        # Select the grid-sampled images
        images = images[im_data.image_id]

        return data, images


class PickKImages(ImageTransform):
    """Simply select K images. This can be performed as a random
    sampling or by simply picking one-every-K images following the
    order in which they come in.
    """
    def __init__(self, k, random=False, replace=False):
        self.k = k
        self.random = random
        self.replace = replace

    def _process(self, data: Data, images: SameSettingImageData):
        # Generate the image indices to select
        if self.random:
            idx = torch.from_numpy(np.random.choice(
                range(images.num_views), size=self.k, replace=self.replace))
        else:
            idx = slice(0, images.num_views, self.k)

        # Select images
        images = images[idx]

        return data, images


class PickImagesFromMappingArea(ImageTransform):
    """Transform to drop images and corresponding mappings based on a
    minimum area ratio mappings should account for and a maximum number
    of images to keep.
    """

    def __init__(self, area_ratio=0.02, n_max=None, n_min=0, use_bbox=False):
        self.area_ratio = area_ratio
        self.n_max = n_max if n_max is not None and n_max >= 1 else None
        self.n_min = n_min if n_max is not None and n_min >= 0 else 0
        self.use_bbox = use_bbox

    def _process(self, data: Data, images: SameSettingImageData):
        assert images.mappings is not None, "No mappings found in images."

        # Threshold below which the number of pixels in the mappings
        # is deemed insufficient
        threshold = images.img_size[0] * images.img_size[1] * self.area_ratio

        # Count the number of pixel mappings for each image
        pixel_idx = images.mappings.images.repeat_interleave(
            images.mappings.values[1].pointers[1:]
            - images.mappings.values[1].pointers[:-1])

        if not self.use_bbox:
            areas = torch_scatter.scatter_add(
                torch.ones(pixel_idx.shape[0]), pixel_idx, dim=0)
        else:
            xy_min = torch_scatter.scatter_min(
                images.mappings.pixels.int(), pixel_idx, dim=0)[0]
            xy_max = torch_scatter.scatter_max(
                images.mappings.pixels.int(), pixel_idx, dim=0)[0]
            x_min = xy_min[:, 0]
            y_min = xy_min[:, 1]
            x_max = xy_max[:, 0]
            y_max = xy_max[:, 1]
            areas = (x_max - x_min) * (y_max - y_min)

        # Compute the indices of image to keep
        n_max = images.num_views if self.n_max is None else self.n_max
        idx = areas.argsort().flip(0)
        idx = idx[areas[idx] > threshold][:n_max]

        # In case no images meet the requirements, pick the one with
        # the largest mapping to avoid having an empty idx
        if idx.shape[0] == 0 and images.num_views > 0 and self.n_min > 0:
            idx = idx[:self.n_min]

        # Select the images and mappings meeting the threshold
        return data, images[idx]


class PickImagesFromMemoryCredit(ImageTransform):
    """Transform to cherry-pick `SameSettingImageData` from an
    `ImageData` object based on an allocated memory credit.
    """

    _PROCESS_IMAGE_DATA = True

    def __init__(self, credit=None, img_size=[], k_coverage=0, n_img=0):
        if credit is not None:
            self.credit = credit
        elif len(img_size) == 2 and n_img > 0:
            self.credit = img_size[0] * img_size[1] * n_img
        else:
            raise ValueError(
                "Either credit or img_size and n_img must be provided.")
        self.use_coverage = k_coverage > 0
        self.k_coverage = k_coverage

    def _process(self, data: Data, images: ImageData):
        # We use lists in favor of arrays or tensors to facilitate
        # item popping

        # Skip if no image
        if images.num_views == 0:
            return data, images

        # Compute the global indexing pair for each image and the list
        # of picked image
        picked = [[] for _ in range(images.num_views)]
        img_indices = [
            [i, j] for i, im in enumerate(images) for j in range(im.num_views)]

        # Compute the image sizes and viewed points boolean masks
        img_sizes = [
            images[i].img_size[0] * images[i].img_size[1]
            for i, j in img_indices]

        # Compute the unseen points boolean masks and split them in a
        # list of masks for easier popping
        if self.use_coverage:
            img_unseen_points = torch.zeros(
                images.num_views, data.num_nodes, dtype=torch.bool)
            i_offset = 0
            for im in images:
                mappings = im.mappings
                i_idx = mappings.images + i_offset
                j_idx = mappings.points.repeat_interleave(
                    mappings.pointers[1:] - mappings.pointers[:-1])
                img_unseen_points[i_idx, j_idx] = True
                i_offset += im.num_views
            img_unseen_points = [x.numpy() for x in img_unseen_points]

        # Credit init
        credit = self.credit

        assert credit > 0 and credit >= min(img_sizes), \
            f"Insufficient credit={credit} to pick any of the provided " \
            f"images with min_size={min(img_sizes)}."

        while credit > 0 and len(img_indices) > 0 and credit >= min(img_sizes):
            # Drop images that are too large to fit the remaining credit
            for idx in range(len(img_indices), 0, -1):
                if img_sizes[idx - 1] > credit:
                    img_indices.pop(idx - 1)
                    img_sizes.pop(idx - 1)
                    if self.use_coverage:
                        img_unseen_points.pop(idx - 1)

            # Compute the coverage factor for each image, defined as
            # the normalized number of yet-unseen points each image
            # carries
            if self.use_coverage:
                w_cov = np.array([x.sum() for x in img_unseen_points])
                w_cov = self.k_coverage * w_cov / (w_cov.max() + 1)
            else:
                w_cov = np.zeros(len(img_indices))

            # Compute the size weights, defined as the normalized
            # pixel area of each image
            w_size = np.array(img_sizes) / np.array(img_sizes).max()

            # Compute the weight of each image, based on its size and
            # unseen points coverage
            weights = w_size + w_cov

            # Normalize the weights into probabilities
            probas = weights / weights.sum()

            # Pick one of the remaining images
            idx = np.random.choice(np.arange(probas.shape[0]), p=probas)

            # Pop the selected image from image attributes
            i, j = img_indices.pop(idx)
            s = img_sizes.pop(idx)
            if self.use_coverage:
                newly_seen = img_unseen_points.pop(idx)

            # Update the picked images, unseen points and credit left
            picked[i].append(j)
            credit -= s
            if self.use_coverage:
                img_unseen_points = [np.logical_and(x, ~newly_seen)
                                     for x in img_unseen_points]

        # Select the images, remove image data if need be
        images = ImageData([im[torch.LongTensor(idx)]
                            for im, idx in zip(images, picked)
                            if len(idx) > 0])

        return data, images


class PickMappingsFromMappingFeatures(ImageTransform):
    """Transform to drop mappings based on mapping features upper or
    lower thresholds.

    Takes as input a list of int (or int) of mapping feature indices,
    optional lists of float (or float) for corresponding lower and upper
    bounds.
    """

    def __init__(self, feat=None, lower=None, upper=None):
        self.feat = self.sanitize(feat)
        self.lower = self.sanitize(lower)
        self.upper = self.sanitize(upper)

        if len(self.lower) == 0:
            self.lower = [None] * len(self.feat)
        if len(self.upper) == 0:
            self.upper = [None] * len(self.feat)

        for x in [self.lower, self.upper]:
            assert len(x) == len(self.feat), \
                f"{x} has {len(x)} elements but {len(self.feat)} were expected."

    @staticmethod
    def sanitize(x):
        if x is None:
            x = []
        elif not isinstance(x, Iterable):
            x = [x]
        return x

    def _process(self, data: Data, images: SameSettingImageData):
        # Skip if no mappings or no mapping features found
        if images.mappings is None or not images.mappings.has_features \
                or len(self.feat) == 0:
            return data, images

        # Check feature indices validity
        assert max(self.feat) == 0 \
               or max(self.feat) < images.mappings.features.shape[1], \
            f"Out of bounds feature id {max(self.feat)}."

        # Iteratively update the view mask
        view_mask = torch.ones(images.mappings.num_items, dtype=torch.bool)
        features = images.mappings.features.view(images.mappings.num_items, -1)
        for i_feat, lower, upper in zip(self.feat, self.lower, self.upper):
            if lower is not None:
                view_mask = view_mask & (features[:, i_feat] > lower)
            if upper is not None:
                view_mask = view_mask & (features[:, i_feat] < upper)

        # Apply the view mask to the images and mappings
        images = images.select_views(view_mask)

        return data, images


class JitterMappingFeatures(ImageTransform):
    """Transform to add a small gaussian noise to the mapping feature.

    Parameters
    ----------
    sigma:
        Variance of the noise
    clip:
        Maximum amplitude of the noise
    """

    def __init__(self, sigma=0.02, clip=0.03):
        self.sigma = sigma
        self.clip = clip

    def _process(self, data: Data, images: SameSettingImageData):
        # Skip if no mappings or no mapping features found
        if images.mappings is None or not images.mappings.has_features:
            return data, images

        # Apply clamped gaussian noise to the mapping features
        noise = self.sigma * torch.randn(images.mappings.features.shape)
        noise = noise.clamp(-self.clip, self.clip)
        images.mappings.features = images.mappings.features + noise

        return data, images


class CenterRoll(ImageTransform):
    """Transform to center the mappings along the width axis of
    spherical images. The images and mappings are rolled along the width
    so as to position the center of the mappings as close to the center
    of the image as possible.

    This assumes the images have a circular representation (ie that the
    first and last pixels along the width are adjacent in reality).

    Does not support prior cropping along the width or resizing.
    """

    def __init__(self, angular_res=16):
        assert isinstance(angular_res, int)
        assert angular_res <= 256
        self.angular_res = angular_res

    def _process(self, data: Data, images: SameSettingImageData):
        # Make sure no prior cropping or resizing was applied to the
        # images and mappings
        assert images.mappings is not None, "No mappings found in images."
        assert images.ref_size[0] == images.img_size[0], \
            f"{self.__class__.__name__} cannot operate if images and " \
            f"mappings underwent prior cropping or resizing."
        assert images.crop_size is None \
               or images.crop_size[0] == images.ref_size[0], \
            f"{self.__class__.__name__} cannot operate if images and " \
            f"mappings underwent prior cropping or resizing."
        assert images.downscale is None or images.downscale == 1, \
            f"{self.__class__.__name__} cannot operate if images and " \
            f"mappings underwent prior cropping or resizing."

        # Skip if no image mappings
        if images.mappings.images.shape[0] == 0:
            return data, images

        # Isolate the mappings pixel widths and associated image ids
        idx = images.mappings.images.repeat_interleave(
            images.mappings.values[1].pointers[1:]
            - images.mappings.values[1].pointers[:-1])
        w_pix = images.mappings.pixels[:, 0]

        # Convert to uint8 and keep unique values
        w_pix = (w_pix.float() * 256 / images.ref_size[0]).long()
        idx, w_pix = lexunique(idx, w_pix)
        w_pix = w_pix.byte()

        # Create the rolled coordinates in a new dimension
        rolls = torch.arange(0, 256, int(256 / self.angular_res)).byte()
        w_pix = torch.cat([(w_pix + r).view(-1, 1) for r in rolls], dim=1)

        # Search for min and max for each roll offset
        w_min, _ = torch_scatter.scatter_min(w_pix, idx, dim=0)
        w_max, _ = torch_scatter.scatter_max(w_pix, idx, dim=0)

        # Compute the centering distance for each roll offset
        w_center_dist = ((w_max.float() + w_min) / 2. - 128).abs().int()

        # Compute the mapping span for each roll offset
        w_span = w_max.int() - w_min

        # Combine center distance and span into a common cost metric
        w_cost = w_span + w_center_dist

        # Search for rollings minimizing centering distances
        idx = torch.arange(w_cost.shape[0])
        roll_idx = w_cost.min(axis=1).indices
        rollings = (rolls[roll_idx] / 256. * images.ref_size[0]).long()
        # Make sure the image ids are preserved
        assert (idx == torch.arange(images.num_views)).all(), \
            "Image indices discrepancy in the rollings."

        # Edit images internal state
        images.update_rollings(rollings)

        return data, images


class CropImageGroups(ImageTransform):
    """Transform to crop images and mappings in a greedy fashion, so as
    to minimize the size of the images while preserving all the mappings
    and padding constraints. This is typically useful for optimizing the
    size of the images to embed with respect to the available mappings.

    The images are distributed to a set of cropping sizes, based on
    their mappings and the padding. Images with the same cropping size
    are batched together.

    Returns an `ImageData` made of `SameSettingImageData` of fixed
    cropping sizes with their respective mappings.
    """

    def __init__(self, padding=0, min_size=64):
        assert padding >= 0, \
            f"Expected a positive scalar but got {padding} instead."
        assert ((min_size & (min_size - 1)) == 0) & (min_size != 0), \
            f"Expected a power of two but got {min_size} instead."
        self.padding = padding
        self.min_size = min_size

    def _process(self, data: Data, images: SameSettingImageData):
        assert images.mappings is not None, "No mappings found in images."

        # If no images, just return an empty ImageData with proper
        # number of points in the mappings. This is important, for it
        # will break `MMBatch.from_mm_data_list` otherwise
        if images.num_views == 0:
            return data, ImageData([images])

        # Compute the bounding boxes for each image
        w_min, w_max, h_min, h_max = images.mappings.bounding_boxes

        # Add padding to the boxes
        w_min = torch.clamp(w_min - self.padding, 0)
        h_min = torch.clamp(h_min - self.padding, 0)
        w_max = torch.clamp(w_max + self.padding, 0, images.img_size[0])
        h_max = torch.clamp(h_max + self.padding, 0, images.img_size[1])
        widths = w_max - w_min
        heights = h_max - h_min

        # Compute the family of possible crop sizes and assign each
        # image to the relevant one.
        # The first size is (min_size, min_size). The other ones follow:
        # (min_size * 2^a, min_size * 2^b)), with a = b or a = b+1. The
        # Size grows this way until both sides reach the full img_size.
        crop_families = {}
        size = (self.min_size, self.min_size)
        i_crop = 0
        image_ids = torch.arange(images.num_views)
        while all(a <= b for a, b in zip(size, images.img_size)):
            if image_ids.shape[0] == 0:
                break

            # Safety measure to make sure all images are used
            if size == tuple(images.img_size):
                crop_families[size] = image_ids
                break

            # Search among the remaining images those that would fit in
            # the crop size
            valid_ids = torch.logical_and(
                widths[image_ids] <= size[0],
                heights[image_ids] <= size[1])

            # Add the image ids to the crop_family of current size
            if image_ids[valid_ids].shape[0] > 0:
                crop_families[size] = image_ids[valid_ids]

            # Discard selected image ids from the remaining image_ids
            image_ids = image_ids[~valid_ids]

            # Compute the next the size. Ensure none of the size sides
            # outsizes img_size
            size = (
                min(size[0] * 2 ** ((i_crop + 1) % 2), images.img_size[0]),
                min(size[1] * 2 ** (i_crop % 2), images.img_size[1]))
            i_crop += 1

        # Make sure the last crop size is the full image
        if images.img_size not in crop_families.keys() \
                and image_ids.shape[0] > 0:
            crop_families[tuple(images.img_size)] = image_ids

        # Index and crop the images and mappings
        for size, idx in crop_families.items():
            # Compute the crop offset for each image. Center mappings
            # inside their cropping boxes while respecting borders.
            off_x = torch.clamp(
                (w_min[idx] - (size[0] - widths[idx]) / 2.).long(),
                0, images.img_size[0] - size[0])
            off_y = torch.clamp(
                (h_min[idx] - (size[1] - heights[idx]) / 2.).long(),
                0, images.img_size[1] - size[1])
            offsets = torch.stack((off_x, off_y), dim=1).long()

            # Index images and mappings and update their cropping
            crop_families[size] = images[idx].update_cropping(size, offsets)

        # Create a holder for the `SameSettingImageData` of each crop size
        return data, ImageData(list(crop_families.values()))


# TODO: CropFromMask
class CropFromMask(ImageTransform):
    """Transform to crop top and bottom from images and mappings based
    on mask.
    """
    pass


# TODO: PadImages
class PadImages(ImageTransform):
    """Transform to update the mappings to account for image padding."""
    #  https://distill.pub/2019/computing-receptive-fields/
    #  https://github.com/google-research/receptive_field
    #  https://github.com/Fangyh09/pytorch-receptive-field
    #  https://github.com/rogertrullo/Receptive-Field-in-Pytorch/blob/master/compute_RF.py
    #  https://fomoro.com/research/article/receptive-field-calculator#3,1,1,SAME;3,1,1,SAME;2,2,1,SAME;3,1,1,SAME;3,1,1,SAME;2,2,1,SAME;3,1,1,SAME;3,1,1,SAME
    pass


class AddPixelHeightFeature(ImageTransform):
    """Transform to add the pixel height to the image features."""

    # TODO: take crop and roll into account to call anytime
    def _process(self, data: Data, images: SameSettingImageData):
        if images.x is None:
            images.load()

        batch, channels, height, width = images.x.shape
        feat = torch.linspace(0, 1, height).float()
        feat = feat.view(1, 1, height, 1).repeat(batch, 1, 1, width)
        images.x = torch.cat((images.x, feat), 1)

        return data, images


class AddPixelWidthFeature(ImageTransform):
    """Transform to add the pixel width to the image features."""

    # TODO: take crop and roll into account to call anytime
    def _process(self, data: Data, images: SameSettingImageData):
        if images.x is None:
            images.load()

        batch, channels, height, width = images.x.shape
        feat = torch.linspace(0, 1, width).float()
        feat = feat.view(1, 1, 1, width).repeat(batch, 1, height, 1)
        images.x = torch.cat((images.x, feat), 1)

        return data, images


class RandomHorizontalFlip(ImageTransform):
    """Horizontally flip the given image randomly with a given
    probability.
    """

    def __init__(self, p=0.50):
        self.p = p

    def _process(self, data: Data, images: SameSettingImageData):
        if images.x is None:
            images.load()

        if torch.rand(1) <= self.p:
            # Turns out that torch.flip is slower than fancy indexing:
            # https://github.com/pytorch/pytorch/issues/16424
            # images.x = torch.flip(images.x, [3])
            flip_index = torch.arange(images.x.shape[-1] - 1, -1, -1)
            images.x = images.x[..., flip_index]

            _, _, _, width = images.x.shape
            images.mappings.pixels[:, 0] = \
                width - 1 - images.mappings.pixels[:, 0]

        return data, images


class ToFloatImage(ImageTransform):
    """Transform to convert [0, 255] uint8 images into [0, 1] float
    tensors.
    """

    def _process(self, data: Data, images: SameSettingImageData):
        if images.x is None:
            images.load()

        images.x = images.x.float() / 255

        return data, images


class TorchvisionTransform(ImageTransform):
    """Torchvision-based transform on the images."""

    def __init__(self):
        raise NotImplementedError

    def _process(self, data: Data, images: SameSettingImageData):
        images.x = self.transform(images.x)
        return data, images

    def __repr__(self):
        return self.transform.__repr__()


class ColorJitter(TorchvisionTransform):
    """Randomly change the brightness, contrast and saturation of an
    image.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.transform = T.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation)


class GaussianBlur(TorchvisionTransform):
    """Blur image with randomly chosen Gaussian blur."""

    def __init__(self, kernel_size=10, sigma=(0.1, 2.0)):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.transform = T.GaussianBlur(kernel_size, sigma=sigma)


class Normalize(TorchvisionTransform):
    """Normalize image colors.

    Default parameters set from ImageNet and ADE20K pretrained models:
    https://github.com/pytorch/vision/issues/39#issuecomment-403701432
    https://github.com/CSAILVision/semantic-segmentation-pytorch.
    """

    def __init__(self,  mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
        self.transform = T.Normalize(mean=mean, std=std)


# TODO : add invertible transforms from https://github.com/gregunz/invertransforms
#  or modify the mappings when applying the geometric transforms.
#  WARNING : if the image undergoes geometric transform, this may cause
#  problems when doing image wrapping or in EquiConv. IDEA : spherical
#  image rotation for augmentation
