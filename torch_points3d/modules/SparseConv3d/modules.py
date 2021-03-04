import torch

import sys

from torch_points3d.utils.config import is_list
from torch_points3d.core.common_modules import Seq, Identity
import torch_points3d.modules.SparseConv3d.nn as snn


class ResBlock(torch.nn.Module):
    """
    Basic ResNet type block

    Parameters
    ----------
    input_nc:
        Number of input channels
    output_nc:
        number of output channels
    convolution
        Either MinkowskConvolution or MinkowskiConvolutionTranspose
    dimension:
        Dimension of the spatial grid
    """

    def __init__(self, input_nc, output_nc, convolution, bias=False):
        super().__init__()
        self.block = (
            Seq()
            .append(convolution(input_nc, output_nc, kernel_size=3, stride=1,
                                bias=bias))
            .append(snn.BatchNorm(output_nc))
            .append(snn.ReLU())
            .append(convolution(output_nc, output_nc, kernel_size=3, stride=1,
                                bias=bias))
            .append(snn.BatchNorm(output_nc))
            .append(snn.ReLU())
        )

        if input_nc != output_nc:
            self.downsample = (
                Seq().append(snn.Conv3d(input_nc, output_nc, kernel_size=1,
                                        stride=1, bias=bias)
                             ).append(snn.BatchNorm(output_nc))
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


class BottleneckBlock(torch.nn.Module):
    """
    Bottleneck block with residual
    """

    def __init__(self, input_nc, output_nc, convolution, reduction=4, bias=False):
        super().__init__()

        self.block = (
            Seq()
            .append(snn.Conv3d(input_nc, output_nc // reduction, kernel_size=1,
                               stride=1, bias=bias))
            .append(snn.BatchNorm(output_nc // reduction))
            .append(snn.ReLU())
            .append(convolution(output_nc // reduction, output_nc // reduction,
                                kernel_size=3, stride=1, bias=bias))
            .append(snn.BatchNorm(output_nc // reduction))
            .append(snn.ReLU())
            .append(snn.Conv3d(output_nc // reduction, output_nc, kernel_size=1,
                               bias=bias))
            .append(snn.BatchNorm(output_nc))
            .append(snn.ReLU())
        )

        if input_nc != output_nc:
            self.downsample = (
                Seq().append(convolution(input_nc, output_nc, kernel_size=1,
                                         stride=1, bias=bias)
                             ).append(snn.BatchNorm(output_nc))
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


class ResNetDown(torch.nn.Module):
    """
    Resnet block that looks like

    in --- strided conv ---- Block ---- sum --[... N times]
                         |              |
                         |-- 1x1 - BN --|
    """

    CONVOLUTION = "Conv3d"

    def __init__(
        self, down_conv_nn=[], kernel_size=2, dilation=1, stride=2, N=1,
            bias=False, block="ResBlock", **kwargs,
    ):
        super().__init__()

        # Recover the block module
        block = getattr(_res_blocks, block)

        # Compute the number of channels for the ResNetDown modules
        nc_in, nc_stride_out, nc_block_in, nc_out = self._parse_conv_nn(
            down_conv_nn, stride, N)

        # Recover the convolution module
        conv = getattr(snn, self.CONVOLUTION)

        # Build the initial strided convolution
        self.conv_in = (
            Seq().append(conv(
                in_channels=nc_in,
                out_channels=nc_stride_out,
                kernel_size=kernel_size,
                stride=stride,
                bias=bias,
                dilation=dilation))
            .append(snn.BatchNorm(nc_stride_out))
            .append(snn.ReLU()))

        # Build the N subsequent blocks
        if N > 0:
            self.blocks = Seq()
            for _ in range(N):
                self.blocks.append(block(nc_block_in, nc_out, conv, bias=bias))
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
        out = self.conv_in(x)
        if self.blocks:
            out = self.blocks(out)
        return out


class ResNetUp(ResNetDown):
    """
    Same as Down conv but for the Decoder
    """

    CONVOLUTION = "Conv3dTranspose"

    def __init__(self, up_conv_nn=[], kernel_size=2, dilation=1, stride=2, N=1,
                 bias=False, skip_first=False, **kwargs):
        self.skip_first = skip_first
        super().__init__(
            down_conv_nn=up_conv_nn, kernel_size=kernel_size, dilation=dilation,
            bias=bias, stride=stride, N=N, **kwargs)

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
                x = snn.cat(x, skip)
            x = self.conv_in(x)
        else:
            x = self.conv_in(x)
            if skip is not None:
                x = snn.cat(x, skip)
        if self.blocks:
            x = self.blocks(x)
        return x
