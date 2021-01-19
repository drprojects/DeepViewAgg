from torch_points3d.core.data_transform.multimodal import instantiate_multimodal_transforms, ComposeMultiModal
from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.utils.config import ConvolutionFormatFactory
import torch_geometric
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
    """
    BaseDataset with multimodal support.
    """

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        BaseDatasetMM.set_multimodal_transform(self, dataset_opt)


    def process(self):
        """
        Instantiate this in child classes because multimodal transforms are 
        very dataset-dependent.
        """
        raise NotImplementedError


    @staticmethod
    def _get_collate_function(conv_type, is_multiscale):
        """
        Make use of torch_geometric Batch mechanisms on '*index*' attributes to
        collate multimodal mapping indices.
        """
        if is_multiscale:
            raise NotImplementedError("Multiscale not supported for multimodal "
                                      "data.")

        is_dense = ConvolutionFormatFactory.check_is_dense_format(conv_type)
        if is_dense:
            raise NotImplementedError("Dense conv_type not supported for "
                                      "multimodal data.")

        # We ake use of the core torch_geometric Batch mechanisms. In
        # particular, '*index*' attributes will be treated carefully
        # when batching. The values are reindexed, which is what we
        # need for our forward star indexing structure.
        return torch_geometric.data.batch.Batch.from_data_list
        

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
