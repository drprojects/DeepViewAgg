from abc import ABC

import torch
import torch.nn as nn
import torchvision
import os.path as osp
import sys
from torch_points3d.utils.config import *
from torch_points3d.core.common_modules import Seq, Identity
from math import pi, sqrt

from mit_semseg.config import cfg as MITCfg
from mit_semseg.models import ModelBuilder as MITModelBuilder
from mit_semseg.lib.nn import SynchronizedBatchNorm2d as MITSynchronizedBatchNorm2d


class ModalityIdentity(Identity):
    """Identiy module for modalities.

    Works just as torch_points3d.core.common_modules.Identity but
    supports unused kwargs in its `__init__` and `forward`.
    """
    def __init__(self, **kwargs):
        super(ModalityIdentity, self).__init__()

    def forward(self, x, **kwargs):
        return x


def standardize_weights(weight, scaled=True):
    weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
        keepdim=True).mean(dim=3, keepdim=True)
    weight = weight - weight_mean
    std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
    fan_in = torch.Tensor([weight.shape[1]]).to(weight.device)
    if scaled:
        # Goes hand-in-hand with ReLUWS to scale the activation output
        weight = weight / (std.expand_as(weight) * torch.sqrt(fan_in))
    else:
        weight = weight / std.expand_as(weight)
    return weight


class Conv2dWS(nn.Conv2d, ABC):
    """Convd2 with weight standardization.

    Sources:
        - https://github.com/joe-siyuan-qiao/WeightStandardization
        - https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html
        - https://arxiv.org/pdf/2102.06171.pdf
        - https://arxiv.org/pdf/1603.01431.pdf
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', scaled=True):
        super(Conv2dWS, self).__init__(in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias, padding_mode=padding_mode)
        self.scaled = scaled

    def forward(self, x, **kwargs):
        weights = standardize_weights(self.weight, scaled=self.scaled)
        return self._conv_forward(x, weights)


class ConvTranspose2dWS(nn.ConvTranspose2d, ABC):
    """Convd2 with weight standardization.

    sources:
        - https://github.com/joe-siyuan-qiao/WeightStandardization
        - https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html
        - https://arxiv.org/pdf/2102.06171.pdf
        - https://arxiv.org/pdf/1603.01431.pdf
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', scaled=True):
        super(ConvTranspose2dWS, self).__init__(in_channels, out_channels,
            kernel_size, stride=stride, padding=padding,
            output_padding=output_padding, dilation=dilation, groups=groups,
            bias=bias, padding_mode=padding_mode)
        self.scaled = scaled

    def forward(self, x, output_size=None, **kwargs):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for '
                             'ConvTranspose2d')

        output_padding = self._output_padding(x, output_size,
            self.stride, self.padding, self.kernel_size)

        weights = standardize_weights(self.weight, scaled=self.scaled)

        return nn.functional.conv_transpose2d(
            x, weights, self.bias, self.stride, self.padding, output_padding,
            self.groups, self.dilation)


class ReLUWS(nn.ReLU, ABC):
    """ReLU with weight standardization.

    sources:
        - https://github.com/joe-siyuan-qiao/WeightStandardization
        - https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html
        - https://arxiv.org/pdf/2102.06171.pdf
        - https://arxiv.org/pdf/1603.01431.pdf
    """
    _SCALE = sqrt(2 / (1 - 1 / pi))

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return nn.functional.relu(input, inplace=self.inplace) * self._SCALE

    def extra_repr(self) -> str:
        return f"inplace={self.inplace}"


class ResBlock(nn.Module, ABC):
    """
    Basic ResNet type block

    Parameters
    ----------
    input_nc:
        Number of input channels
    output_nc:
        number of output channels
    convolution
        Either Conv2d or ConvTranspose2d
    """

    # TODO: extend to EquiConv https://github.com/palver7/EquiConvPytorch

    def __init__(self, input_nc, output_nc, convolution, normalization,
                 activation):
        if convolution in [nn.ConvTranspose2d, ConvTranspose2dWS]:
            padding_mode = 'zeros'
        else:
            padding_mode = 'reflect'

        super().__init__()
        self.block = (
            Seq().append(convolution(
                input_nc,
                output_nc,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode=padding_mode))
            .append(normalization(output_nc))
            .append(activation())
            .append(convolution(
                output_nc,
                output_nc,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode=padding_mode))
            .append(normalization(output_nc))
            .append(activation()))

        if input_nc != output_nc:
            self.downsample = (
                Seq().append(nn.Conv2d(
                    input_nc,
                    output_nc,
                    kernel_size=1,
                    stride=1))
                .append(normalization(output_nc)))
        else:
            self.downsample = None

    def forward(self, x, **kwargs):
        out = self.block(x)
        if self.downsample:
            out += self.downsample(x)
        else:
            out += x
        return out


class BottleneckBlock(nn.Module, ABC):
    """Bottleneck block with residual."""

    def __init__(self, input_nc, output_nc, convolution, normalization,
                 activation, reduction=4, **kwargs):
        super().__init__()

        if convolution in [nn.ConvTranspose2d, ConvTranspose2dWS]:
            padding_mode = 'zeros'
        else:
            padding_mode = 'reflect'

        self.block = (
            Seq().append(convolution(
                input_nc,
                output_nc // reduction,
                kernel_size=1,
                stride=1))
            .append(normalization(output_nc // reduction))
            .append(activation())
            .append(convolution(
                output_nc // reduction,
                output_nc // reduction,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode=padding_mode))
            .append(normalization(output_nc // reduction))
            .append(activation())
            .append(convolution(
                output_nc // reduction,
                output_nc,
                kernel_size=1,))
            .append(normalization(output_nc))
            .append(activation()))

        if input_nc != output_nc:
            self.downsample = (
                Seq().append(convolution(
                    input_nc,
                    output_nc,
                    kernel_size=1,
                    stride=1))
                .append(normalization(output_nc)))
        else:
            self.downsample = None

    def forward(self, x, **kwargs):
        out = self.block(x)
        if self.downsample:
            out += self.downsample(x)
        else:
            out += x
        return out


_local_modules = sys.modules[__name__]


class ResNetDown(nn.Module, ABC):
    """
    Resnet block that looks like

    in --- strided conv ---- Block ---- sum --[... N times]
                         |              |
                         |-- 1x1 - BN --|
    """

    CONVOLUTION = "Conv2d"
    ACTIVATION = "ReLU"

    def __init__(
            self, down_conv_nn=[], kernel_size=2, dilation=1, stride=2, N=1,
            padding=0, block="ResBlock", padding_mode='reflect',
            normalization='BatchNorm2d', weight_standardization=False,
            **kwargs):
        super().__init__()

        # If an empty down_conv_nn or channel sizes smaller than 1 are
        # passed, the ResNetDown will simply become a pass-through
        # Identity module
        if len(down_conv_nn) < 2 or any([x < 0 for x in down_conv_nn]):
            self.conv_in = None
            self.blocks = None
            return

        # Recover the block module
        block = getattr(_local_modules, block)

        # Compute the number of channels for the ResNetDown modules
        nc_in, nc_stride_out, nc_block_in, nc_out = self._parse_conv_nn(
            down_conv_nn, stride, N)

        # Recover the convolution and activation modules
        if weight_standardization:
            conv = getattr(_local_modules, self.CONVOLUTION + 'WS')
            activation = getattr(_local_modules, self.ACTIVATION + 'WS')
        else:
            conv = getattr(nn, self.CONVOLUTION)
            activation = getattr(nn, self.ACTIVATION)

        # Recover the normalization module from torch.nn, for GroupNorm
        # the number of groups is set to distribute ~16 channels per
        # group: https://arxiv.org/pdf/1803.08494.pdf
        if normalization == 'GroupNorm':
            norm = lambda nc: nn.GroupNorm(max(nc // 16, 1), nc)
        else:
            norm = getattr(nn, normalization)

        # Build the initial strided convolution
        self.conv_in = (
            Seq().append(conv(
                in_channels=nc_in,
                out_channels=nc_stride_out,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                padding_mode=padding_mode))
            .append(norm(nc_stride_out))
            .append(activation()))

        # Build the N subsequent blocks
        if N > 0:
            self.blocks = Seq()
            for _ in range(N):
                self.blocks.append(block(nc_block_in, nc_out, conv,
                     norm, activation))
                nc_block_in = nc_out
        else:
            self.blocks = None

    def _parse_conv_nn(self, down_conv_nn, stride, N):
        if is_list(down_conv_nn[0]):
            down_conv_nn = down_conv_nn[0]
        assert len(down_conv_nn) == 2, \
            f"ResNetDown expects down_conv_nn to have length of 2 to carry " \
            f"(nc_in, nc_out) but got len(down_conv_nn)={len(down_conv_nn)}."
        nc_in, nc_out = down_conv_nn
        nc_stride_out = nc_in if stride > 1 and N > 0 else nc_out
        nc_block_in = nc_stride_out
        return nc_in, nc_stride_out, nc_block_in, nc_out

    def forward(self, x, **kwargs):
        if self.conv_in:
            x = self.conv_in(x)
        if self.blocks:
            x = self.blocks(x)
        return x


class ResNetUp(ResNetDown, ABC):
    """Same as ResNetDown but for the Decoder."""

    CONVOLUTION = "ConvTranspose2d"

    def __init__(self, up_conv_nn=[], kernel_size=2, dilation=1, stride=2, N=1,
                 padding=0, padding_mode='zeros', normalization='BatchNorm2d',
                 weight_standardization=False, skip_first=False, **kwargs):
        self.skip_first = skip_first
        super().__init__(
            down_conv_nn=up_conv_nn, kernel_size=kernel_size,
            dilation=dilation, stride=stride, N=N, padding=padding,
            padding_mode=padding_mode, normalization=normalization,
            weight_standardization=weight_standardization, **kwargs)

    def _parse_conv_nn(self, up_conv_nn, stride, N):
        if is_list(up_conv_nn[0]):
            up_conv_nn = up_conv_nn[0]

        if self.skip_first:
            assert len(up_conv_nn) == 2, \
                f"ResNetUp with skip_first=True expects down_conv_nn to have " \
                f"length of 2 to carry (nc_in, nc_out) but got " \
                f"len(up_conv_nn)={len(up_conv_nn)}."
        else:
            assert len(up_conv_nn) == 3, \
                f"ResNetUp with skip_first=False expects up_conv_nn to have " \
                f"length of 3 to carry (nc_in, nc_skip_in, nc_out) but got " \
                f"len(up_conv_nn)={len(up_conv_nn)}."

        if self.skip_first:
            nc_in, nc_out = up_conv_nn
            nc_stride_out = nc_in if stride > 1 and N > 0 else nc_out
            nc_block_in = nc_stride_out
        else:
            nc_in, nc_skip_in, nc_out = up_conv_nn
            nc_stride_out = nc_in if stride > 1 and N > 0 else nc_out
            nc_block_in = nc_stride_out + nc_skip_in

        return nc_in, nc_stride_out, nc_block_in, nc_out

    def forward(self, x, skip, **kwargs):
        if self.skip_first:
            if skip is not None:
                x = torch.cat((x, skip), dim=1)
            if self.conv_in:
                x = self.conv_in(x)
        else:
            if self.conv_in:
                x = self.conv_in(x)
            if skip is not None:
                x = torch.cat((x, skip), dim=1)
        if self.blocks:
            x = self.blocks(x)
        return x

    def extra_repr(self) -> str:
        return f"skip_first={self.skip_first}"


class UnaryConv(nn.Module, ABC):
    """1x1 convolution on image."""

    def __init__(self, input_nc, output_nc, normalization=None, activation=None,
                 weight_standardization=False, in_drop=0, out_drop=0,
                 persistent_drop=False):
        super().__init__()
        # Build the input Dropout if any
        if in_drop is None or in_drop <= 0:
            self.input_dropout = None
        elif persistent_drop:
            self.input_dropout = PersistentDropout2d(input_nc, p=in_drop)
        else:
            self.input_dropout = Dropout2d(p=in_drop, inplace=True)

        # Recover the normalization module from torch.nn, for GroupNorm
        # the number of groups is set to distribute ~16 channels per
        # group: https://arxiv.org/pdf/1803.08494.pdf
        if normalization is None:
            self.norm = None
        elif normalization == 'GroupNorm':
            self.norm = lambda nc: nn.GroupNorm(max(nc // 16, 1), nc)
        else:
            self.norm = getattr(nn, normalization)

        # Build the 1x1 convolution and activation
        if weight_standardization:
            self.conv = Conv2dWS(input_nc, output_nc, stride=1, kernel_size=1)
            self.activation = getattr(_local_modules, activation + 'WS') \
                if activation is not None else None
        else:
            self.conv = nn.Conv2d(input_nc, output_nc, stride=1, kernel_size=1)
            self.activation = getattr(nn, activation) \
                if activation is not None else None

        # Build the output Dropout if any
        if out_drop is None or out_drop <= 0:
            self.output_dropout = None
        elif persistent_drop:
            self.output_dropout = PersistentDropout2d(output_nc, p=out_drop)
        else:
            self.output_dropout = Dropout2d(p=out_drop, inplace=True)

    def forward(self, x, **kwargs):
        if self.input_dropout:
            x = self.input_dropout(x, **kwargs)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        if self.output_dropout:
            x = self.output_dropout(x, **kwargs)
        return x


class Dropout2d(nn.Dropout2d):
    """ Dropout2d with kwargs support. """
    def forward(self, input, **kwargs):
        return super().forward(input)


class PersistentDropout2d(nn.Module):
    """ Dropout2d with a persistent dropout mask.
    The mask may be reset at froward time. This is useful when the same
    Dropout mask needs to be applied to various input batches.

    Inspired from:
    https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/lock_dropout.html

    Args:
        input_nc (int): Number of input channels.
        p (float): Probability of an element in the dropout mask to be zeroed.
    """
    def __init__(self, input_nc, p=0.5):
        self.input_nc = input_nc
        self.p = p
        self.mask = None
        super().__init__()

    def forward(self, x, reset=True, **kwargs):
        """
        Args:
            x :class:`torch.FloatTensor`: Input to apply dropout too.
            reset (bool, optional): If set to ``True``, will reset the
                dropout mask.
        """
        # Dropout acts as Identity in eval mode
        if not self.training or not self.p:
            self.mask = None
            return x

        # Reset the feature dropout mask
        if self.mask is None or reset:
            mask = x.new_empty(1, self.input_nc, 1, 1, requires_grad=False
                               ).bernoulli_(1 - self.p)
            self.mask = mask.div_(1 - self.p)

        return x * self.mask.expand_as(x)

    def extra_repr(self) -> str:
        return f"input_nc={self.input_nc}, p={self.p}"


SPECIAL_NAMES = ["block_names"]


class UNet(nn.Module, ABC):
    """Generic UNet module for images.

    Create the Unet from a dictionary of compact options.

    For each part of the architecture, the blocks are implicitly
    pre-selected:
      - Down  : ResNetDown
      - Inner : BottleneckBlock
      - Up    : ResNetUp

    opt is expected to have the following format:
            down_conv:
                down_conv_nn: ...
                *args

            innermost: [OPTIONAL]
                *args

            up_conv:
                up_conv_nn: ...
                *args

    Inspired from torch_points3d/models/base_architectures/unet.py
    """

    def __init__(self, opt):
        super().__init__()

        # Detect which options format has been used to define the model
        if is_list(opt.down_conv) or "down_conv_nn" not in opt.down_conv \
                or is_list(opt.up_conv) or 'up_conv_nn' not in opt.up_conv:
            raise NotImplementedError
        else:
            self._init_from_compact_format(opt)

    def _init_from_compact_format(self, opt):
        # Down modules
        self.down_modules = nn.ModuleList()
        for i in range(len(opt.down_conv.down_conv_nn)):
            down_module = self._build_module(opt.down_conv, i, "DOWN")
            self.down_modules.append(down_module)

        # Innermost module
        if hasattr(opt, "innermost") and opt.innermost is not None:
            self.inner_modules = nn.ModuleList()
            inners = self._build_module(opt.innermost, 0, "INNER")
            self.inner_modules.append(inners)
        else:
            self.inner_modules = None

        # Up modules
        self.up_modules = nn.ModuleList()
        for i in range(len(opt.up_conv.up_conv_nn)):
            up_module = self._build_module(opt.up_conv, i, "UP")
            self.up_modules.append(up_module)

        # Final 1x1 conv
        if hasattr(opt, "last_conv") and opt.last_conv is not None:
            last = self._build_module(opt.last_conv, 0, "LAST")
            self.last = last
        else:
            self.last = None

    def _build_module(self, opt, index, flow):
        """Builds a convolution (up, down or inner) block.

        Arguments:
            conv_opt - model config subset describing the convolutional
                block
            index - layer index in sequential order (as they come in the
                config)
            flow - UP, DOWN or INNER
        """
        if flow.lower() == 'DOWN'.lower():
            module_cls = ResNetDown
        elif flow.lower() == 'INNER'.lower():
            module_cls = BottleneckBlock
        elif flow.lower() == 'UP'.lower():
            module_cls = ResNetUp
        elif flow.lower() == 'LAST'.lower():
            module_cls = UnaryConv
        else:
            raise NotImplementedError
        args = fetch_arguments_from_list(opt, index, SPECIAL_NAMES)
        return module_cls(**args)

    def forward(self, x, **kwargs):
        """This method does a forward on the Unet assuming symmetrical
        skip connections.

        Parameters
        ----------
        x: torch.Tensor of images [BxCxHxW]
        """
        stack_down = []
        for i in range(len(self.down_modules) - 1):
            x = self.down_modules[i](x)
            stack_down.append(x)
        x = self.down_modules[-1](x)

        if self.inner_modules is not None:
            raise NotImplementedError
            # TODO: debug innermost, stacks and upconv
            x = self.inner_modules[0](x)

        # Recover the skip mode from the up modules
        if self.up_modules[0].skip_first:
            stack_down.append(None)

        for i in range(len(self.up_modules)):
            skip = stack_down.pop(-1) if stack_down else None
            x = self.up_modules[i](x, skip)

        if self.last is not None:
            x = self.last(x, **kwargs)

        return x


class PrudentSynchronizedBatchNorm2d(MITSynchronizedBatchNorm2d):
    """MITSynchronizedBatchNorm2d with support for (1, C, 1, 1) inputs at
    training time.
    """

    def forward(self, input, **kwargs):
        is_training = self.training
        if input.shape[0] == input.shape[2] == input.shape[3] == 1:
            self.training = False
        output = super(PrudentSynchronizedBatchNorm2d, self).forward(input)
        self.training = is_training
        return output

    @classmethod
    def from_pretrained(cls, bn_pretrained):
        # Initialize to default PPMFeatMap instance
        bn_new = cls(bn_pretrained.num_features)

        # Recover all attributes
        for k, v in bn_pretrained.__dict__.items():
            setattr(bn_new, k, v)

        return bn_new


class PPMFeatMap(nn.Module):
    """Pyramid Pooling Module for feature extraction.

    Adapted from https://github.com/CSAILVision/semantic-segmentation-pytorch
    """
    def __init__(self, fc_dim=4096, pool_scales=(1, 2, 3, 6)):
        super(PPMFeatMap, self).__init__()

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                PrudentSynchronizedBatchNorm2d(512),  # (1, C, 1, 1) inputs hurt basic BN
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim + len(pool_scales) * 512, 512,
                      kernel_size=3, padding=1, bias=False),
            MITSynchronizedBatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

    @classmethod
    def from_pretrained(cls, ppm_pretrained):
        # Initialize to default PPMFeatMap instance
        ppm_new = cls()

        # Recover the PPM module
        ppm_new.ppm = ppm_pretrained.ppm

        # Change the PPM MITSynchronizedBatchNorm2d to PrudentBatchNorm2d
        # to handle single-image batches
        for m in ppm_new.ppm:
            m[2] = PrudentSynchronizedBatchNorm2d.from_pretrained(m[2])

        # Recover the conv_last module without dropout and classifier
        ppm_new.conv_last = nn.Sequential(*list(ppm_pretrained.conv_last)[:-2])
        return ppm_new

    def forward(self, conv_out, out_size=None, **kwargs):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if out_size is not None:
            x = nn.functional.interpolate(
                x, size=out_size, mode='bilinear', align_corners=False)

        return x


class ADE20KResNet18PPM(nn.Module, ABC):
    """ResNet-18 encoder with PPM decoder pretrained on ADE20K.

    Adapted from https://github.com/CSAILVision/semantic-segmentation-pytorch
    """

    def __init__(self, *args, frozen=False, **kwargs):
        super(ADE20KResNet18PPM, self).__init__()

        # Adapt the default config to use ResNet18 + PPM-Deepsup model
        ARCH = 'resnet18dilated-ppm_deepsup'
        DIR = osp.join(
            osp.dirname(osp.abspath(__file__)), 'pretrained/ade20k', ARCH)
        MITCfg.merge_from_file(osp.join(DIR, f'{ARCH}.yaml'))
        MITCfg.MODEL.arch_encoder = MITCfg.MODEL.arch_encoder.lower()
        MITCfg.MODEL.arch_decoder = MITCfg.MODEL.arch_decoder.lower()
        MITCfg.DIR = DIR

        # Absolute paths of model weights
        MITCfg.MODEL.weights_encoder = osp.join(
            MITCfg.DIR, 'encoder_' + MITCfg.TEST.checkpoint)
        MITCfg.MODEL.weights_decoder = osp.join(
            MITCfg.DIR, 'decoder_' + MITCfg.TEST.checkpoint)

        assert osp.exists(MITCfg.MODEL.weights_encoder) and \
               osp.exists(MITCfg.MODEL.weights_decoder), \
            "checkpoint does not exist!"

        # Build encoder and decoder from pretrained weights
        self.encoder = MITModelBuilder.build_encoder(
            arch=MITCfg.MODEL.arch_encoder,
            fc_dim=MITCfg.MODEL.fc_dim,
            weights=MITCfg.MODEL.weights_encoder)
        self.decoder = MITModelBuilder.build_decoder(
            arch=MITCfg.MODEL.arch_decoder,
            fc_dim=MITCfg.MODEL.fc_dim,
            num_class=MITCfg.DATASET.num_class,
            weights=MITCfg.MODEL.weights_decoder,
            use_softmax=True)

        # Convert PPM from a classifier into a feature map extractor
        self.decoder = PPMFeatMap.from_pretrained(self.decoder)

        # If the model is frozen, it will always remain in eval mode
        # and the parameters will have requires_grad=False
        self.frozen = frozen
        if self.frozen:
            self.training = False

    def forward(self, x, out_size=None, **kwargs):
        pred = self.decoder(self.encoder(x, return_feature_maps=True),
            out_size=out_size)
        return pred

    @property
    def frozen(self):
        return self._frozen

    @frozen.setter
    def frozen(self, frozen):
        if isinstance(frozen, bool):
            self._frozen = frozen
        for p in self.parameters():
            p.requires_grad = not self.frozen

    def train(self, mode=True):
        return super(ADE20KResNet18PPM, self).train(mode and not self.frozen)


class ADE20KResNet18TruncatedLayer4(nn.Module):
    """ResNet-18 encoder pretrained on ADE20K with PPM decoder.

    Adapted from https://github.com/CSAILVision/semantic-segmentation-pytorch
    """
    _LAYERS = ['layer0', 'layer1', 'layer2', 'layer3', 'layer4']
    _LAYERS_IN = {k: v for k, v in zip(_LAYERS, [3, 128, 64, 128, 256])}
    _LAYERS_OUT = {k: v for k, v in zip(_LAYERS, [128, 64, 128, 256, 512])}

    def __init__(self, frozen=False, **kwargs):
        super(ADE20KResNet18TruncatedLayer4, self).__init__()

        # Adapt the default config to use ResNet18 + PPM-Deepsup model
        ARCH = 'resnet18dilated-ppm_deepsup'
        DIR = osp.join(
            osp.dirname(osp.abspath(__file__)), 'pretrained/ade20k', ARCH)
        MITCfg.merge_from_file(osp.join(DIR, f'{ARCH}.yaml'))
        MITCfg.MODEL.arch_encoder = MITCfg.MODEL.arch_encoder.lower()
        MITCfg.DIR = DIR

        # Absolute paths of model weights
        MITCfg.MODEL.weights_encoder = osp.join(
            MITCfg.DIR, 'encoder_' + MITCfg.TEST.checkpoint)

        assert osp.exists(MITCfg.MODEL.weights_encoder), \
            "checkpoint does not exist!"

        # Build encoder from pretrained weights
        resnet18 = MITModelBuilder.build_encoder(
            arch=MITCfg.MODEL.arch_encoder,
            fc_dim=MITCfg.MODEL.fc_dim,
            weights=MITCfg.MODEL.weights_encoder)

        # Combine the ResNet first conv-bn-relu blocks and maxpool as
        # layer0
        resnet18.layer0 = nn.Sequential(
            resnet18.conv1,
            resnet18.bn1,
            resnet18.relu1,
            resnet18.conv2,
            resnet18.bn2,
            resnet18.relu2,
            resnet18.conv3,
            resnet18.bn3,
            resnet18.relu3,
            resnet18.maxpool)

        # Combine the selected layers into a nn.Sequential
        self.conv = nn.Sequential(
            *[getattr(resnet18, layer) for layer in self._LAYERS])

        # If the model is frozen, it will always remain in eval mode
        # and the parameters will have requires_grad=False
        self.frozen = frozen
        if self.frozen:
            self.training = False

    def forward(self, x, **kwargs):
        return self.conv(x)

    @property
    def input_nc(self):
        return self._LAYERS_IN[self._LAYERS[0]]

    @property
    def output_nc(self):
        return self._LAYERS_OUT[self._LAYERS[-1]]

    @property
    def frozen(self):
        return self._frozen

    @frozen.setter
    def frozen(self, frozen):
        if isinstance(frozen, bool):
            self._frozen = frozen
        for p in self.parameters():
            p.requires_grad = not self.frozen

    def train(self, mode=True):
        return super(ADE20KResNet18TruncatedLayer4, self).train(
            mode and not self.frozen)


class ADE20KResNet18TruncatedLayer0(ADE20KResNet18TruncatedLayer4):
    _LAYERS = ['layer0']


class ADE20KResNet18TruncatedLayer1(ADE20KResNet18TruncatedLayer4):
    _LAYERS = ['layer0', 'layer1']


class ADE20KResNet18TruncatedLayer2(ADE20KResNet18TruncatedLayer4):
    _LAYERS = ['layer0', 'layer1', 'layer2']


class ADE20KResNet18TruncatedLayer3(ADE20KResNet18TruncatedLayer4):
    _LAYERS = ['layer0', 'layer1', 'layer2', 'layer3']


class ADE20KResNet18Layer0(ADE20KResNet18TruncatedLayer4):
    _LAYERS = ['layer0']


class ADE20KResNet18Layer1(ADE20KResNet18TruncatedLayer4):
    _LAYERS = ['layer1']


class ADE20KResNet18Layer2(ADE20KResNet18TruncatedLayer4):
    _LAYERS = ['layer2']


class ADE20KResNet18Layer3(ADE20KResNet18TruncatedLayer4):
    _LAYERS = ['layer3']


class ADE20KResNet18Layer4(ADE20KResNet18TruncatedLayer4):
    _LAYERS = ['layer4']


def _instantiate_torchvision_resnet(
        arch, block, layers, pretrained, progress, **kwargs):
    """Instantiate ResNet models from torchvision, optionally
    pretrained on ImageNet. Supported models are 'resnet18', 'resnet34',
    'resnet50', 'resnet101' and 'resnet152'.

    This is a custom version of torchvision.models.resnet._resnet to
    support locally-saved pretrained ResNet weights.
    """
    model = torchvision.models.resnet.ResNet(block, layers)

    if pretrained:

        model_dir = osp.join(
            osp.dirname(osp.abspath(__file__)), f'pretrained/imagenet/{arch}')
        file_name = f'{arch}.pth'
        file_path = osp.join(model_dir, file_name)

        # Load from local weights
        if osp.exists(file_path):
            state_dict = torch.load(file_path)

        # Load ImageNet-pretrained weights from official torchvision URL
        # and save them locally
        else:
            url = torchvision.models.resnet.model_urls[arch]
            state_dict = torchvision.models.utils.load_state_dict_from_url(
                url, progress=progress, model_dir=model_dir,
                file_name=file_name)

        model.load_state_dict(state_dict)
    return model


class ResNet18TruncatedLayer4(nn.Module):
    _LAYERS = ['layer0', 'layer1', 'layer2', 'layer3', 'layer4']
    _LAYERS_IN = {k: v for k, v in zip(_LAYERS, [3, 64, 64, 128, 256])}
    _LAYERS_OUT = {k: v for k, v in zip(_LAYERS, [64, 64, 128, 256, 512])}

    def __init__(self, frozen=False, pretrained=True, **kwargs):
        super(ResNet18TruncatedLayer4, self).__init__()

        # Instantiate the full ResNet
        resnet18 = _instantiate_torchvision_resnet(
            'resnet18', torchvision.models.resnet.BasicBlock, [2, 2, 2, 2],
            pretrained, True, **kwargs)

        # Combine the ResNet first conv1-bn1-relu-maxpool as layer0
        resnet18.layer0 = nn.Sequential(
            resnet18.conv1, resnet18.bn1, resnet18.relu, resnet18.maxpool)

        # Combine the selected layers into a nn.Sequential
        self.conv = nn.Sequential(
            *[getattr(resnet18, layer) for layer in self._LAYERS])

        # If the model is frozen, it will always remain in eval mode
        # and the parameters will have requires_grad=False
        self.frozen = frozen
        if self.frozen:
            self.training = False

    def forward(self, x, **kwargs):
        return self.conv(x)

    @property
    def input_nc(self):
        return self._LAYERS_IN[self._LAYERS[0]]

    @property
    def output_nc(self):
        return self._LAYERS_OUT[self._LAYERS[-1]]

    @property
    def frozen(self):
        return self._frozen

    @frozen.setter
    def frozen(self, frozen):
        if isinstance(frozen, bool):
            self._frozen = frozen
        for p in self.parameters():
            p.requires_grad = not self.frozen

    def train(self, mode=True):
        return super(ResNet18TruncatedLayer4, self).train(
            mode and not self.frozen)


class ResNet18TruncatedLayer0(ResNet18TruncatedLayer4):
    _LAYERS = ['layer0']


class ResNet18TruncatedLayer1(ResNet18TruncatedLayer4):
    _LAYERS = ['layer0', 'layer1']


class ResNet18TruncatedLayer2(ResNet18TruncatedLayer4):
    _LAYERS = ['layer0', 'layer1', 'layer2']


class ResNet18TruncatedLayer3(ResNet18TruncatedLayer4):
    _LAYERS = ['layer0', 'layer1', 'layer2', 'layer3']


class ResNet18Layer0(ResNet18TruncatedLayer4):
    _LAYERS = ['layer0']


class ResNet18Layer1(ResNet18TruncatedLayer4):
    _LAYERS = ['layer1']


class ResNet18Layer2(ResNet18TruncatedLayer4):
    _LAYERS = ['layer2']


class ResNet18Layer3(ResNet18TruncatedLayer4):
    _LAYERS = ['layer3']


class ResNet18Layer4(ResNet18TruncatedLayer4):
    _LAYERS = ['layer4']
