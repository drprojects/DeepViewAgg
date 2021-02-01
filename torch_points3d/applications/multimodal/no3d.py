import logging
import torch
from torch_points3d.models.base_architectures.unet import UnwrappedUnetBasedModel
from torch_points3d.applications.utils import extract_output_nc
from torch_points3d.core.common_modules.base_modules import MLP
from torch_points3d.datasets.multimodal.data import MMData
from torch_geometric.data import Data, Batch

log = logging.getLogger(__name__)


class No3DEncoder(UnwrappedUnetBasedModel):
    """Encoder structure for multimodal models without 3D data.

    Inspired from torchpoints_3d.applications.sparseconv3d.
    """

    def __init__(self, model_config, model_type, dataset, modules, *args, **kwargs):
        # UnwrappedUnetBasedModel init
        super(No3DEncoder, self).__init__(model_config, model_type, dataset, modules)

        # Make sure the model is multimodal and has no 3D. Note that
        # the UnwrappedUnetBasedModel carries most of the required
        # initialization.
        assert self.is_multimodal and self.no_3d, \
            f"No3DUnet should carry at least one non-3D modality."
        assert self.no_3d, f"No3DUnet should not have 3D-specific modules."

        # BN and transpose conv weights init
        self.weight_initialization()

        # Recover size of output features
        default_output_nc = kwargs.get("default_output_nc", None)
        if not default_output_nc:
            mod_out_nc_list = [extract_output_nc(getattr(model_config, m))
                               for m in self.modalities]
            assert all(o == mod_out_nc_list[0] for o in mod_out_nc_list), \
                f"Expected all modality branches outputs to have the same " \
                f"feature size but got {mod_out_nc_list} sizes instead."
            default_output_nc = mod_out_nc_list[0]
        self._output_nc = default_output_nc

        # Set the MLP head if any
        self._has_mlp_head = False
        if "output_nc" in kwargs:
            self._has_mlp_head = True
            self._output_nc = kwargs["output_nc"]
            self.mlp = MLP([default_output_nc, self.output_nc],
                           activation=torch.nn.ReLU(), bias=False)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) \
                    or isinstance(m, torch.nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.kernel, mode="fan_out",
                                              nonlinearity="relu")

            if isinstance(m, torch.nn.BatchNorm):
                torch.nn.init.constant_(m.bn.weight, 1)
                torch.nn.init.constant_(m.bn.bias, 0)

    @property
    def has_mlp_head(self):
        return self._has_mlp_head

    @property
    def output_nc(self):
        return self._output_nc

    def _set_input(self, data: MMData):
        """Unpack input data from the dataloader and perform necessary
        pre-processing steps.

        Parameters
        -----------
        data: MMData object
        """
        self.input = (None, data.to(self.device).modalities)
        if data.pos is not None:
            self.xyz = data.pos

    def forward(self, data, *args, **kwargs):
        """Run forward pass. Expects a MMData object for input, with
        3D Data and multimodal data and mappings. Although the
        No3DEncoder model does not apply any convolution modules
        directly on the 3D points, it still requires a 3D points Data
        object with a 'pos' attribute as input, to be able to output
        these very same points populated with modality-generated
        features.

        Parameters
        -----------
        data: MMData object

        Returns
        --------
        data: Data object
            - pos [N, 3] (coords or real pos if xyz is in data)
            - x [N, output_nc]
        """
        self._set_input(data)
        data = self.input
        for i in range(len(self.down_modules)):
            data = self.down_modules[i](data)

        # Discard the modalities used in the down modules, only
        # 3D point features are expected to be used in subsequent
        # modules. Restore the input Data object equipped with the
        # proper point positions and modality-generated features.
        out = Batch(x=data[0], pos=self.xyz).to(self.device)

        # Apply the MLP head, if any
        if self.has_mlp_head:
            out.x = self.mlp(out.x)

        return out
