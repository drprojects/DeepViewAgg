from torch_points3d.core.data_transform.multimodal import instantiate_multimodal_transforms, ComposeMultiModal
from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.core.multimodal.data import MMBatch
from torch_points3d.utils.config import ConvolutionFormatFactory
from functools import partial
import copy


def explode_multimodal_transform(transforms):
    """ Returns a flattened list of transform
    Arguments:
        transforms {[list | ComposeMultiModal]} -- Contains list of transform
        to be added

    Returns:
        [list] -- [List of transforms]
    """
    out = []
    if transforms is not None:
        if isinstance(transforms, ComposeMultiModal):
            out = copy.deepcopy(transforms.transforms)
        elif isinstance(transforms, list):
            out = copy.deepcopy(transforms)
        else:
            raise Exception("Multimodal transforms should be provided either "
                            "within a list or a ComposeMultiModal")
    return out


class BaseDatasetMM(BaseDataset):
    """BaseDataset with multimodal support.
    """

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        BaseDatasetMM.set_multimodal_transform(self, dataset_opt)

    def process(self):
        """Instantiate this in child classes because multimodal
        transforms are very dataset-dependent.
        """
        raise NotImplementedError

    @staticmethod
    def _get_collate_function(conv_type, is_multiscale,
                              pre_collate_transform=None):
        """Collate mechanism for MMData.

        Relies on MMBatch.from_mm_data_list to preserve multimodal
        mappings and features when.
        """
        if is_multiscale:
            raise NotImplementedError(
                "Multiscale not supported for multimodal data.")

        is_dense = ConvolutionFormatFactory.check_is_dense_format(conv_type)
        if is_dense:
            raise NotImplementedError(
                "Dense conv_type not supported for multimodal data.")

        # We ake use of the core torch_geometric Batch mechanisms. In
        # particular, '*index*' attributes will be treated carefully
        # when batching. The values are reindexed, which is what we
        # need for our forward star indexing structure.
        fn = MMBatch.from_mm_data_list
        return partial(BaseDataset._collate_fn, collate_fn=fn,
                       pre_collate_transform=pre_collate_transform)

    @staticmethod
    def remove_multimodal_transform(transform_in, list_transform_class):
        """Remove a multimodal transform if within list_transform_class

        Arguments:
            transform_in {[type]} -- [ComposeMultiModal | List of transform]
            list_transform_class {[type]} -- [List of transform class to be removed]

        Returns:
            [type] -- [description]
        """
        if isinstance(transform_in, ComposeMultiModal) or isinstance(transform_in, list):
            if len(list_transform_class) > 0:
                transform_out = []
                transforms = transform_in.transforms if isinstance(transform_in, ComposeMultiModal) else transform_in
                for t in transforms:
                    if not isinstance(t, tuple(list_transform_class)):
                        transform_out.append(t)
                transform_out = ComposeMultiModal(transform_out)
        else:
            transform_out = transform_in
        return transform_out

    @staticmethod
    def set_multimodal_transform(obj, dataset_opt):
        """This function creates and sets the method used for multimodal 
        mapping, based on the configuration multimodal attributes in the passed
        configuration.

        Inspired from BaseDataset.set_transform().
        """
        for k in dataset_opt.keys():
            if k != "multimodal":
                continue

            # Recover the modality name and options
            modality_opt = getattr(dataset_opt, k)
            modality = getattr(modality_opt, 'modality')

            # Initialize the modality transforms to None
            for prefix in ['pre', 'test', 'train', 'val']:
                setattr(obj, f"{prefix}_transform_{modality}", None)

            for key in modality_opt.keys():
                if "transform" not in key:
                    continue

                transform = instantiate_multimodal_transforms(
                    getattr(modality_opt, key))

                setattr(
                    obj,
                    f"{key.replace('transforms', 'transform')}_{modality}",
                    transform)

            # Chain pre_transform_modality and test_transform_modality
            # in inference_transform_modality
            inference_transform = explode_multimodal_transform(
                getattr(obj, f"pre_transform_{modality}"))
            inference_transform += explode_multimodal_transform(
                getattr(obj, f"test_transform_{modality}"))
            inference_transform = ComposeMultiModal(inference_transform) \
                if len(inference_transform) > 0 else None
            setattr(obj, f"inference_transform_{modality}", inference_transform)
