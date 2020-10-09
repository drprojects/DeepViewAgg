from torch_points3d.core.data_transform import instantiate_multimodal_transform



class BaseDatasetMM(BaseDataset):
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        BaseDatasetMM.set_multimodal_transform(self, dataset_opt)


    def process(self):
        """
        Instantiate this in child classes because multimodal transforms are 
        very dataset-dependent.
        """
        if is_multiscale:
            raise NotImplementedError("Multiscale not supported for multimodal data.")

        is_dense = ConvolutionFormatFactory.check_is_dense_format(conv_type)
        if is_dense:
            raise NotImplementedError("Dense conv_type not supported for multimodal data.")

        # We ake use of the core torch_geometric Batch mechanisms
        # In particular, '*index*' attributes will be treated carefully when 
        # batching. The values are reindexed, which is what we need for our
        # forward star indexing structure. 
        return torch_geometric.data.batch.Batch.from_data_list


    @staticmethod
    def _get_collate_function(conv_type, is_multiscale):
        """Dedicated collate to capture the batch indices ?"""
        raise NotImplementedError


    @staticmethod
    def set_multimodal_transform(obj, dataset_opt):
        """This function creates and sets the method used for multimodal 
        mapping, based on the configuration multimodal attributes in the passed
        configuration.

        Inspired from BaseDataset.set_transform().
        """

        for k in dataset_opt.keys():
            if k == "multimodal":

                modality_opt = getattr(dataset_opt, k)
                modality = getattr(modality_opt, 'modality')

                for key in modality_opt.keys():
                    if "transform" in key:

                        # Expects only one transform

                        getattr(modality_opt, key)

                        # NB : only the first 'transform' attribute is taken
                        # into account. Multimodal composition is not implemented yet
                        modality_transform = getattr(modality_opt, key)
                        assert len(modality_transform) == 1,
                            "Multimodal composition not implemented."
                        assert hasattr(modality_transform, 'transform'),
                            "No transform found in the configuration."

                        transform = instantiate_multimodal_transform[modality_transform[0]]

                        setattr(obj, f"{key}_{modality}", transform)
