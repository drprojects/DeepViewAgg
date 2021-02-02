import copy
from abc import ABC
from torch import nn
from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.models.base_architectures import BaseFactory, \
    ModalityFactory, get_factory
from torch_points3d.models.base_model import BaseModel
from torch_points3d.modules.multimodal.modules import MultimodalBlockDown, \
    UnimodalBranch
from torch_points3d.utils.config import is_list, get_from_kwargs, \
    fetch_arguments_from_list, flatten_compact_options, fetch_modalities
from torch_points3d.datasets.multimodal.data import MODALITY_NAMES
import logging

log = logging.getLogger(__name__)

SPECIAL_NAMES = ["radius", "max_num_neighbors", "block_names"]


# --------------------------- BACKBONE BASE -------------------------- #

class BackboneBasedModel(BaseModel, ABC):
    """
    create a backbone-based generator:
    This is simply an encoder
    (can be used in classification, regression, metric learning and so one)
    """

    def __init__(self, opt, model_type, dataset: BaseDataset, modules_lib):

        """Construct a backbone generator (It is a simple down module)
        Parameters:
            opt - options for the network generation
            model_type - type of the model to be generated
            modules_lib - all modules that can be used in the backbone


        opt is expected to contains the following keys:
        * down_conv
        """
        opt = copy.deepcopy(opt)
        super(BackboneBasedModel, self).__init__(opt)
        self._spatial_ops_dict = {"neighbour_finder": [], "sampler": []}

        # Check if one of the supported modalities is present in the config
        self._modalities = fetch_modalities(opt.down_conv, MODALITY_NAMES)

        # Check if the 3D convolutions are specified in the config
        self.no_3d_conv = "down_conv_nn" not in opt.down_conv

        # Detect which options format has been used to define the model
        if is_list(opt.down_conv) or self.no_3d_conv and not self.is_multimodal:
            raise NotImplementedError
        else:
            self._init_from_compact_format(opt, model_type, dataset, modules_lib)

    def _init_from_compact_format(self, opt, model_type, dataset, modules_lib):
        """Create a backbonebasedmodel from the compact options format
        - where the same convolution is given for each layer, and
        arguments are given in lists.
        """
        # Initialize the down module list
        self.down_modules = nn.ModuleList()

        # Factory for creating down modules for the main 3D modality
        factory_module_cls = get_factory(model_type, modules_lib)
        down_conv_cls_name = opt.down_conv.module_name
        self._module_factories = {'main': factory_module_cls(
            down_conv_cls_name, None, modules_lib)}

        # Factories for creating modules for additional modalities
        if self.is_multimodal:
            for m in self.modalities:
                mod_opt = getattr(opt.down_conv, m)
                self._module_factories[m] = ModalityFactory(
                    m,
                    mod_opt.down_conv.module_name,
                    mod_opt.atomic_pooling.module_name,
                    mod_opt.view_pooling.module_name,
                    mod_opt.fusion.module_name)

        # Down modules - 3D conv only
        down_modules = []
        if self.no_3d_conv:
            down_modules = [nn.Identity()] * 2
        else:
            for i in range(len(opt.down_conv.down_conv_nn)):
                down_conv_3d = self._build_module(opt.down_conv, i, flow="DOWN")
                self._save_sampling_and_search(down_conv_3d)
                down_modules.append(down_conv_3d)

        # Down modules - modality-specific branches
        if self.is_multimodal:
            assert len(down_modules) % 2 == 0 and len(down_modules) > 0, \
                f"Expected an even number of 3D conv modules but got " \
                f"{len(down_modules)} modules instead."
            n_layers_down = len(down_modules) // 2

            branches = [{m: nn.Identity()
                         for m in self.modalities}] * n_layers_down

            for m in self.modalities:
                # Get the branching indices
                b_idx = getattr(opt.down_conv, m).branching_index
                b_idx = [b_idx] if not is_list(b_idx) else b_idx

                # Check whether the modality module is a UNet
                is_unet = hasattr(getattr(opt.down_conv, m), 'up_conv') \
                          and getattr(opt.down_conv, m).up_conv is not None
                assert not is_unet or len(b_idx) == 1, \
                    f"Cannot build a {m}-specific UNet with multiple " \
                    f"branching indices. Consider removing the 'up_conv' " \
                    f"from the {m} modality or providing a single branching " \
                    f"index."

                # Ensure the modality has only one branching index if
                # the model has no 3D down conv
                assert not self.no_3d_conv or b_idx == [0], \
                    f"Cannot build a no-3D model with multiple " \
                    f"branching indices. All modality-specific branches " \
                    f"should join at the index=0. Consider changing the" \
                    f"branching index of the {m} modality to 0 or building " \
                    f"3D convolutions."

                # Build the branches
                for i, idx in enumerate(b_idx):
                    if is_unet:
                        unet_cls = self._module_factories[m].get_module('UNET')
                        conv = unet_cls(getattr(opt.down_conv, m))
                    else:
                        conv = self._build_module(
                            getattr(opt.down_conv, m).down_conv, i, modality=m)
                    atomic_pool = self._build_module(
                        getattr(opt.down_conv, m).atomic_pooling, i, modality=m,
                        flow='ATOMIC')
                    view_pool = self._build_module(
                        getattr(opt.down_conv, m).view_pooling, i, modality=m,
                        flow='VIEW')
                    fusion = self._build_module(
                        getattr(opt.down_conv, m).fusion, i, modality=m,
                        flow='FUSION')

                    # Group modules into a UnimodalBranch
                    branch = UnimodalBranch(conv, atomic_pool, view_pool, fusion)

                    # Update the branches at the proper branching point
                    branches[idx][m] = branch

            # Update the down_modules list
            down_modules = [
                MultimodalBlockDown(conv_1, conv_2, **modal_conv)
                for conv_1, conv_2, modal_conv
                in zip(down_modules[::2], down_modules[1::2], branches)]

        # Down modules - combined
        self.down_modules = nn.ModuleList(down_modules)

        self.metric_loss_module, self.miner_module = BaseModel.get_metric_loss_and_miner(
            getattr(opt, "metric_loss", None), getattr(opt, "miner", None)
        )

    def _save_sampling_and_search(self, down_conv):
        sampler = getattr(down_conv, "sampler", None)
        if is_list(sampler):
            self._spatial_ops_dict["sampler"] = sampler + self._spatial_ops_dict["sampler"]
        else:
            self._spatial_ops_dict["sampler"] = [sampler] + self._spatial_ops_dict["sampler"]

        neighbour_finder = getattr(down_conv, "neighbour_finder", None)
        if is_list(neighbour_finder):
            self._spatial_ops_dict["neighbour_finder"] = neighbour_finder + self._spatial_ops_dict["neighbour_finder"]
        else:
            self._spatial_ops_dict["neighbour_finder"] = [neighbour_finder] + self._spatial_ops_dict["neighbour_finder"]


    def _build_module(self, conv_opt, index, flow='DOWN', modality='main'):
        """Builds a convolution (up or down) or a merge block in the case of
        multimodal models.

        Arguments:
            conv_opt - model config subset describing the convolutional block
            index - layer index in sequential order (as they come in the config)
            flow - "UP", "DOWN", "ATOMIC, "VIEW" or "FUSION"
            modality - string among supported modalities
        """
        args = fetch_arguments_from_list(conv_opt, index, SPECIAL_NAMES)
        args["index"] = index
        module = self._module_factories[modality].get_module(flow)
        return module(**args)
