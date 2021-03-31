from abc import ABC

import torch.nn as nn
import sys
from torch_points3d.utils.config import *
from torch_points3d.core.common_modules import Seq, Identity


def standardize_weights(weight, scaled=False):
    weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
        keepdim=True).mean(dim=3, keepdim=True)
    weight = weight - weight_mean
    std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
    fan_in = torch.Tensor([weight.shape[1]]).to(weight.device)
    if scaled:
        weight = weight / (std.expand_as(weight) * torch.sqrt(fan_in))
        # ALSO ... scale ReLU by sqrt(2 / (1 - 1/pi)) ...
    else:
        weight = weight / std.expand_as(weight)
    return weight


class Conv2dWS(nn.Conv2d, ABC):
    """Convd2 with weight standardization.

    Sources:
        - https://github.com/joe-siyuan-qiao/WeightStandardization
        - https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', scaled=False):
        super(Conv2dWS, self).__init__(in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias, padding_mode=padding_mode)
        self.scaled = scaled

    def forward(self, x):
        weights = standardize_weights(self.weight, scaled=self.scaled)
        return self._conv_forward(x, weights)


class ConvTranspose2dWS(nn.ConvTranspose2d, ABC):
    """Convd2 with weight standardization.

    source: https://github.com/joe-siyuan-qiao/WeightStandardization
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', scaled=False):
        super(ConvTranspose2dWS, self).__init__(in_channels, out_channels,
            kernel_size, stride=stride, padding=padding,
            output_padding=output_padding, dilation=dilation, groups=groups,
            bias=bias, padding_mode=padding_mode)
        self.scaled = scaled

    def forward(self, x, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for '
                             'ConvTranspose2d')

        output_padding = self._output_padding(x, output_size,
            self.stride, self.padding, self.kernel_size)

        weights = standardize_weights(self.weight, scaled=self.scaled)

        return torch.nn.functional.conv_transpose2d(
            x, weights, self.bias, self.stride, self.padding, output_padding,
            self.groups, self.dilation)


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

    def __init__(self, input_nc, output_nc, convolution, normalization):
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
            .append(nn.ReLU())
            .append(convolution(
                output_nc,
                output_nc,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode=padding_mode))
            .append(normalization(output_nc))
            .append(nn.ReLU()))

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

    def forward(self, x):
        out = self.block(x)
        if self.downsample:
            out += self.downsample(x)
        else:
            out += x
        return out


class BottleneckBlock(nn.Module, ABC):
    """Bottleneck block with residual."""

    def __init__(self, input_nc, output_nc, convolution, normalization,
                 reduction=4, **kwargs):
        super().__init__()

        if convolution in [nn.ConvTranspose2d, ConvTranspose2dWS]:
            padding_mode = 'zeros'
        else:
            padding_mode = 'reflect'

        self.block = (
            Seq().append(nn.Conv2d(
                input_nc,
                output_nc // reduction,
                kernel_size=1,
                stride=1))
            .append(normalization(output_nc // reduction))
            .append(nn.ReLU())
            .append(convolution(
                output_nc // reduction,
                output_nc // reduction,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode=padding_mode))
            .append(normalization(output_nc // reduction))
            .append(nn.ReLU())
            .append(nn.Conv2d(
                output_nc // reduction,
                output_nc,
                kernel_size=1,))
            .append(normalization(output_nc))
            .append(nn.ReLU()))

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

    def forward(self, x):
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

    def __init__(
            self, down_conv_nn=[], kernel_size=2, dilation=1, stride=2, N=1,
            padding=0, block="ResBlock", padding_mode='reflect',
            normalization='BatchNorm2d', weight_standardization=False,
            **kwargs):
        super().__init__()

        # If an empty down_conv_nn or channel sizes smaller than 1 are
        # passed, the ResNetDown will simply become a pass-through
        # Identity module
        if len(down_conv_nn) < 2 or any([x < 1 for x in down_conv_nn]):
            self.conv_in = None
            self.blocks = None
            return

        # Recover the block module
        block = getattr(_local_modules, block)

        # Compute the number of channels for the ResNetDown modules
        nc_in, nc_stride_out, nc_block_in, nc_out = self._parse_conv_nn(
            down_conv_nn, stride, N)

        # Recover the convolution module
        if weight_standardization:
            conv = getattr(_local_modules, self.CONVOLUTION + 'WS')
        else:
            conv = getattr(nn, self.CONVOLUTION)

        # Recover the normalization module from torch.nn, for GroupNorm
        # the number of groups is set to distribute ~16 channels per
        # group: https://arxiv.org/pdf/1803.08494.pdf
        if normalization == 'GroupNorm':
            norm = lambda nc: nn.GroupNorm(nc // 16, nc)
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
            .append(nn.ReLU()))

        # Build the N subsequent blocks
        if N > 0:
            self.blocks = Seq()
            for _ in range(N):
                self.blocks.append(block(nc_block_in, nc_out, conv,
                     norm))
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

    def forward(self, x):
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

    def forward(self, x, skip):
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


class UnaryConv(nn.Module, ABC):
    """1x1 convolution on image."""

    def __init__(self, input_nc, output_nc, normalization=None, activation=None,
                 weight_standardization=False):
        super().__init__()
        # Recover the normalization module from torch.nn, for GroupNorm
        # the number of groups is set to distribute ~16 channels per
        # group: https://arxiv.org/pdf/1803.08494.pdf
        if normalization is None:
            self.norm = None
        elif normalization == 'GroupNorm':
            self.norm = lambda nc: nn.GroupNorm(nc // 16, nc)
        else:
            self.norm = getattr(nn, normalization)

        # Recover the activation module from torch.nn
        self.activation = getattr(nn, activation) if activation is not None \
            else activation

        # Build the 1x1 convolution
        if weight_standardization:
            self.conv = Conv2dWS(input_nc, output_nc, stride=1, kernel_size=1)
        else:
            self.conv = nn.Conv2d(input_nc, output_nc, stride=1, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


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
            x = self.last(x)

        return x
