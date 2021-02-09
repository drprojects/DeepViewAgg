from abc import ABC

import torch.nn as nn
import sys
from torch_points3d.utils.config import *
from torch_points3d.core.common_modules import Seq, Identity


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

    def __init__(self, input_nc, output_nc, convolution):
        if convolution is nn.ConvTranspose2d:
            padding_mode = 'zeros'
        else:
            padding_mode = 'reflect'

        super().__init__()
        self.block = (
            Seq()
            .append(convolution(input_nc, output_nc, kernel_size=3, stride=1,
                                padding=1, padding_mode=padding_mode))
            .append(nn.BatchNorm2d(output_nc))
            .append(nn.ReLU())
            .append(convolution(output_nc, output_nc, kernel_size=3, stride=1,
                                padding=1, padding_mode=padding_mode))
            .append(nn.BatchNorm2d(output_nc))
            .append(nn.ReLU())
        )

        if input_nc != output_nc:
            self.downsample = (
                Seq().append(
                    nn.Conv2d(input_nc, output_nc, kernel_size=1, stride=1)
                ).append(nn.BatchNorm2d(output_nc))
            )
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
    """
    Bottleneck block with residual
    """

    def __init__(self, input_nc, output_nc, convolution, reduction=4, **kwargs):
        super().__init__()

        if convolution is nn.ConvTranspose2d:
            padding_mode = 'zeros'
        else:
            padding_mode = 'reflect'

        self.block = (
            Seq()
            .append(nn.Conv2d(input_nc, output_nc // reduction, kernel_size=1,
                              stride=1))
            .append(nn.BatchNorm2d(output_nc // reduction))
            .append(nn.ReLU())
            .append(convolution(output_nc // reduction, output_nc // reduction,
                                kernel_size=3, stride=1, padding=1,
                                padding_mode=padding_mode))
            .append(nn.BatchNorm2d(output_nc // reduction))
            .append(nn.ReLU())
            .append(nn.Conv2d(output_nc // reduction, output_nc,
                              kernel_size=1,))
            .append(nn.BatchNorm2d(output_nc))
            .append(nn.ReLU())
        )

        if input_nc != output_nc:
            self.downsample = (
                Seq().append(
                    convolution(input_nc, output_nc, kernel_size=1, stride=1)
                ).append(nn.BatchNorm2d(output_nc))
            )
        else:
            self.downsample = None

    def forward(self, x):
        out = self.block(x)
        if self.downsample:
            out += self.downsample(x)
        else:
            out += x
        return out


_res_blocks = sys.modules[__name__]


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
            padding=0, block="ResBlock", padding_mode='reflect', **kwargs,
    ):
        block = getattr(_res_blocks, block)
        super().__init__()
        if stride > 1:
            conv1_output = down_conv_nn[0]
        else:
            conv1_output = down_conv_nn[1]

        conv = getattr(nn, self.CONVOLUTION)
        self.conv_in = (
            Seq()
            .append(
                conv(
                    in_channels=down_conv_nn[0],
                    out_channels=conv1_output,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=padding,
                    padding_mode=padding_mode,
                )
            )
            .append(nn.BatchNorm2d(conv1_output))
            .append(nn.ReLU())
        )

        if N > 0:
            self.blocks = Seq()
            for _ in range(N):
                self.blocks.append(block(conv1_output, down_conv_nn[1], conv))
                conv1_output = down_conv_nn[1]
        else:
            self.blocks = None

    def forward(self, x):
        x = self.conv_in(x)
        if self.blocks:
            x = self.blocks(x)
        return x


class ResNetUp(ResNetDown, ABC):
    """
    Same as Down conv but for the Decoder
    """

    CONVOLUTION = "ConvTranspose2d"

    def __init__(self, up_conv_nn=[], kernel_size=2, dilation=1, stride=2, N=1,
                 padding=0, padding_mode='zeros', **kwargs):
        super().__init__(
            down_conv_nn=up_conv_nn, kernel_size=kernel_size,
            dilation=dilation, stride=stride, N=N, padding=padding,
            padding_mode=padding_mode, **kwargs,
        )

    def forward(self, x, skip):
        if skip is not None:
            x = torch.cat((x, skip), dim=1)
        return super().forward(x)


class UnaryConv(nn.Module, ABC):
    """1x1 convolution on image."""

    def __init__(self, input_nc, output_nc, norm=None, activation=None):
        super().__init__()
        self.norm = getattr(nn, norm) if norm is not None else norm
        self.activation = getattr(nn, activation) if activation is not None \
            else activation
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
        stack_down.append(None)

        if self.inner_modules is not None:
            raise NotImplementedError
            # TODO: debug innermost, stacks and upconv
            stack_down.append(x)
            x = self.inner_modules[0](x)

        for i in range(len(self.up_modules)):
            x = self.up_modules[i](x, stack_down.pop())

        if self.last is not None:
            x = self.last(x)

        return x
