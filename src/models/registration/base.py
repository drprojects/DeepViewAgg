import logging
import torch.nn.functional as F
from typing import Any
import etw_pytorch_utils as pt_utils
from src.datasets.multiscale_data import MultiScaleBatch
from src.models.base_architectures import BackboneBasedModel

log = logging.getLogger(__name__)


def create_batch_siamese(pair, batch):
    """
    create a batch with siamese input
    """
    return 2 * batch + pair


class PatchSiamese(BackboneBasedModel):
    def __init__(self, option, model_type, dataset, modules):
        """
        Initialize this model class
        Parameters:
            opt -- training/test options
        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """

        BackboneBasedModel.__init__(self, option, model_type, dataset, modules)
        self.set_last_mlp(option.mlp_cls)
        self.loss_names = ["loss_reg"]

    def set_input(self, data):
        self.input = data
        # TODO multiscale data pre_computed...
        if isinstance(data, MultiScaleBatch):
            self.pre_computed = data.multiscale
            del data.multiscale
        else:
            self.pre_computed = None
        # batch siamese
        self.batch_idx = create_batch_siamese(data.pair, data.batch)

    def set_last_mlp(self, last_mlp_opt):
        self.FC_layer = pt_utils.Seq(last_mlp_opt.nn[0])
        for i in range(1, len(last_mlp_opt.nn)):
            self.FC_layer.conv1d(last_mlp_opt.nn[i], bn=True)

    def set_loss(self):
        raise NotImplementedError("Choose a loss for the metric learning")

    def forward(self) -> Any:
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""

        data = self.input
        for i in range(len(self.down_modules)):
            data = self.down_modules[i](data, pre_computed=self.pre_computed)

        x = F.relu(self.lin1(data.x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        self.output = self.lin2(x)

        self.loss_reg = self.loss_module(self.output) + self.get_internal_loss()
        return self.output

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        self.loss_reg.backward()  # calculate gradients of network G w.r.t. loss_G
