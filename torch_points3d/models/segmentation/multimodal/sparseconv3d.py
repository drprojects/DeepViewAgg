from torch_points3d.models.segmentation.sparseconv3d import *
from torch_points3d.models.segmentation.multimodal.no3d import *
from torch_points3d.applications.multimodal.no3d import No3DEncoder
from torch_points3d.utils.model_building_utils.model_definition_resolver \
    import resolve_model
from torch_points3d.modules.SparseConv3d.modules import ResBlock


log = logging.getLogger(__name__)


class LateFeatureFusion(APIModel):
    """Pair of SparseConv3D backbone and a modality-specific backbone
    with late feature fusion mechanism.
    """

    MODES = ['residual', 'concatenation', 'both']

    def __init__(self, option, model_type, dataset, modules):
        # call the initialization method of UnetBasedModel
        BaseModel.__init__(self, option)

        # Resolve the config constants here because instantiate_model
        # fails to parse backbone_3d and backbone_no3d configs
        resolve_model(option.backbone_3d, dataset, None)
        resolve_model(option.backbone_no3d, dataset, None)

        # 3D backbone init
        self.backbone_3d = SparseConv3d(
            "unet", dataset.feature_dimension, config=option.backbone_3d,
            backend=option.get("backend", "minkowski"))

        # No3D backbone init
        self.backbone_no3d = No3DEncoder(
            option.backbone_no3d, model_type, dataset, modules)

        # Set modalities based on the No3D backbone
        self._modalities = self.backbone_no3d.modalities

        # Recover the late fusion mode
        self.mode = option.mode \
            if hasattr(option, 'mode') and option.mode is not None \
            else 'residual'

        # Build the feature fusion head
        if self.mode == 'residual':
            assert self.backbone_3d.output_nc == self.backbone_no3d.output_nc, \
                f"Backbones output dimensions must be the same. Received " \
                f"backbone_3d.output_nc={self.backbone_3d.output_nc} and " \
                f"backbone_no3d.output_nc={self.backbone_no3d.output_nc} " \
                f"instead."
            self.fusion = lambda a, b: a + b
            # self.head = nn.Sequential(nn.Linear(
            #     self.backbone_3d.output_nc, dataset.num_classes))
        elif self.mode == 'concatenation':
            self.fusion = lambda a, b: torch.cat((a, b), dim=-1)
            # self.head = nn.Sequential(nn.Linear(
            #     self.backbone_3d.output_nc + self.backbone_no3d.output_nc,
            #     dataset.num_classes))
        elif self.mode == 'both':
            assert self.backbone_3d.output_nc == self.backbone_no3d.output_nc, \
                f"Backbones output dimensions must be the same. Received " \
                f"backbone_3d.output_nc={self.backbone_3d.output_nc} and " \
                f"backbone_no3d.output_nc={self.backbone_no3d.output_nc} " \
                f"instead."
            self.fusion = lambda a, b: torch.cat((a, a + b), dim=-1)
            # self.head = nn.Sequential(nn.Linear(
            #     self.backbone_3d.output_nc + self.backbone_no3d.output_nc,
            #     dataset.num_classes))
        else:
            raise NotImplementedError(
                f"Unknown fusion mode='{self.mode}'. Please choose among "
                f"supported modes: {self.MODES}.")

        self.late_conv = ResBlock(
            self.backbone_3d.output_nc + self.backbone_no3d.output_nc, 96,
            sp3d.nn.Conv3d)

        self.head = nn.Linear(96, dataset.num_classes)
#         self.head = nn.Linear(self.backbone_3d.output_nc + self.backbone_no3d.output_nc, dataset.num_classes)

        self.loss_names = ["loss_seg"]

        if option.get("loss_no3d", False):
            self.head_no3d = nn.Sequential(nn.Linear(
                self.backbone_no3d.output_nc, dataset.num_classes))
        else:
            self.head_no3d = None

        if option.get("loss_3d", False):
            self.head_3d = nn.Sequential(nn.Linear(
                self.backbone_3d.output_nc, dataset.num_classes))
        else:
            self.head_3d = None

        self._use_cross_entropy = option.get('use_cross_entropy', True)
        self._use_lovasz = option.get('use_lovasz', False)
        assert self._use_cross_entropy or self._use_lovasz, \
            "Choose at least one between Cross-Entropy loss and Lovasz loss."
        self.loss_names = ['loss_seg'] \
                          + self._use_cross_entropy * ['loss_cross_entropy'] \
                          + self._use_lovasz * ['loss_lovasz'] \
                          + (self._use_cross_entropy and self.head_no3d is not None) * ['loss_cross_entropy_no3d'] \
                          + (self._use_cross_entropy and self.head_3d is not None) * ['loss_cross_entropy_3d'] \
                          + (self._use_lovasz and self.head_no3d is not None) * ['loss_lovasz_no3d'] \
                          + (self._use_lovasz and self.head_3d is not None) * ['loss_lovasz_3d']

    def forward(self, *args, **kwargs):
        # 3D backbone
        features_3d = self.backbone_3d(self.input.data).x

        # Need to pass self.input equipped with features_3d to
        # modality-specific backbone, but would prefer not cloning the
        # input batch. So just temporarily change the self.input.data.x
        # and restore it later on.
        input_x_backup = self.input.data.x
        self.input.data.x = features_3d

        # Modality-specific backbone
        features_no3d = self.backbone_no3d(self.input).x

        # Restore self.input.data.x
        self.input.data.x = input_x_backup

#         features = self.fusion(features_3d, features_no3d)
        multimodal_features = self.fusion(features_3d, features_no3d)

        # DIRTY: get sparse tensor with proper coords and pass it multimodal features
        sparse_features = self.backbone_3d.last_sparse_tensor
        sparse_features.F = multimodal_features

        # DIRTY: apply 3D convs to spatially mix multimodal data a bit
        features = self.late_conv(sparse_features).F

        logits = self.head(features)
        self.output = F.log_softmax(logits, dim=-1)

        if self.labels is not None:
            self.loss_seg = 0

            if self.head_no3d:
                logits_no3d = self.head_no3d(features_no3d)
                output_no3d = F.log_softmax(logits_no3d, dim=-1)
                loss_no3d = F.nll_loss(output_no3d, self.labels, ignore_index=IGNORE_LABEL)
                self.loss_seg += loss_no3d

            if self.head_3d:
                logits_3d = self.head_3d(features_3d)
                output_3d = F.log_softmax(logits_3d, dim=-1)
                loss_3d = F.nll_loss(output_3d, self.labels, ignore_index=IGNORE_LABEL)
                self.loss_seg += loss_3d

            if self._use_cross_entropy:
                self.loss_cross_entropy = F.nll_loss(
                    self.output, self.labels, ignore_index=IGNORE_LABEL, weight=getattr(self, '_weight_classes', None))
                self.loss_seg += self.loss_cross_entropy

            if self._use_lovasz:
                self.loss_lovasz = lovasz_softmax(self.output.exp(), self.labels, ignore=IGNORE_LABEL)
                self.loss_seg += self.loss_lovasz

            if self.head_no3d:
                logits_no3d = self.head_no3d(features_no3d)
                output_no3d = F.log_softmax(logits_no3d, dim=-1)
                if self._use_cross_entropy:
                    self.loss_cross_entropy_no3d = F.nll_loss(output_no3d, self.labels, ignore_index=IGNORE_LABEL)
                    self.loss_seg += self.loss_cross_entropy_no3d
                if self._use_lovasz:
                    self.loss_lovasz_no3d = lovasz_softmax(output_no3d.exp(), self.labels, ignore=IGNORE_LABEL)
                    self.loss_seg += self.loss_lovasz_no3d

            if self.head_3d:
                logits_3d = self.head_3d(features_3d)
                output_3d = F.log_softmax(logits_3d, dim=-1)
                if self._use_cross_entropy:
                    self.loss_cross_entropy_3d = F.nll_loss(output_3d, self.labels, ignore_index=IGNORE_LABEL)
                    self.loss_seg += self.loss_cross_entropy_3d
                if self._use_lovasz:
                    self.loss_lovasz_3d = lovasz_softmax(output_3d.exp(), self.labels, ignore=IGNORE_LABEL)
                    self.loss_seg += self.loss_lovasz_3d



class LateLogitFusion(APIModel):
    """Pair of SparseConv3D backbone and a modality-specific backbone
    with late logit fusion mechanism.
    """

    def __init__(self, option, model_type, dataset, modules):
        # call the initialization method of UnetBasedModel
        BaseModel.__init__(self, option)

        # Resolve the config constants here because instantiate_model
        # fails to parse backbone_3d and backbone_no3d configs
        resolve_model(option.backbone_3d, dataset, None)
        resolve_model(option.backbone_no3d, dataset, None)

        # 3D backbone init
        self.backbone_3d = SparseConv3d(
            "unet", dataset.feature_dimension, config=option.backbone_3d,
            backend=option.get("backend", "minkowski"))

        # No3D backbone init
        self.backbone_no3d = No3DEncoder(
            option.backbone_no3d, model_type, dataset, modules)

        # Set modalities based on the No3D backbone
        self._modalities = self.backbone_no3d.modalities

        # Build the logit fusion heads
        # Note: the No3D head is expected to be already set in the
        # No3DEncoder. See 'last_conv' config for No3D architectures.
        self.head_3d = nn.Sequential(nn.Linear(
            self.backbone_3d.output_nc, dataset.num_classes))

        self.loss_names = ["loss_seg"]

    def forward(self, *args, **kwargs):
        features_3d = self.backbone_3d(self.input.data).x
        logits_3d = self.head_3d(features_3d)
        logits_no3d = self.backbone_no3d(self.input).x
        logits = logits_3d + logits_no3d
        self.output = F.log_softmax(logits, dim=-1)
        if self.labels is not None:
            self.loss_seg = F.nll_loss(
                self.output, self.labels, ignore_index=IGNORE_LABEL)
