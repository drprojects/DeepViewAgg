import torch.nn as nn
import sys

from torch_points3d.core.common_modules import Seq, Identity

class ResBlock(nn.Module):
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
        TODO : look into from torchvision.ops.deform_conv to wrap spherical images and improve border features
        TODO : extend to EquiConv https://github.com/palver7/EquiConvPytorch
        TODO : optional maxpool with SegNet structure : https://github.com/say4n/pytorch-segnet/blob/master/src/model.py
    """

    def __init__(self, input_nc, output_nc, convolution):
        super().__init__()
        self.block = (
            Seq()
            .append(convolution(input_nc, output_nc, kernel_size=3, stride=1))
            .append(nn.BatchNorm2d(output_nc))
            .append(nn.ReLU())
            .append(convolution(output_nc, output_nc, kernel_size=3, stride=1))
            .append(nn.BatchNorm2d(output_nc))
            .append(nn.ReLU())
        )

        if input_nc != output_nc:
            self.downsample = (
                Seq().append(nn.Conv2d(input_nc, output_nc, kernel_size=1, stride=1)).append(nn.BatchNorm2d(output_nc))
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


class BottleneckBlock(nn.Module):
    """
    Bottleneck block with residual
    """

    def __init__(self, input_nc, output_nc, convolution, reduction=4):
        super().__init__()

        self.block = (
            Seq()
            .append(nn.Conv2d(input_nc, output_nc // reduction, kernel_size=1, stride=1))
            .append(nn.BatchNorm2d(output_nc // reduction))
            .append(nn.ReLU())
            .append(convolution(output_nc // reduction, output_nc // reduction, kernel_size=3, stride=1,))
            .append(nn.BatchNorm2d(output_nc // reduction))
            .append(nn.ReLU())
            .append(nn.Conv2d(output_nc // reduction, output_nc, kernel_size=1,))
            .append(nn.BatchNorm2d(output_nc))
            .append(nn.ReLU())
        )

        if input_nc != output_nc:
            self.downsample = (
                Seq().append(convolution(input_nc, output_nc, kernel_size=1, stride=1)).append(nn.BatchNorm2d(output_nc))
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


class ResNetDown(nn.Module):
    """
    Resnet block that looks like

    in --- strided conv ---- Block ---- sum --[... N times]
                         |              |
                         |-- 1x1 - BN --|
    """

    CONVOLUTION = "Conv2d"

    def __init__(
        self, down_conv_nn=[], kernel_size=2, dilation=1, stride=2, N=1, block="ResBlock", **kwargs,
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
        out = self.conv_in(x)
        if self.blocks:
            out = self.blocks(out)
        return out


class ResNetUp(ResNetDown):
    """
    Same as Down conv but for the Decoder
    """

    CONVOLUTION = "ConvTranspose2d"

    def __init__(self, up_conv_nn=[], kernel_size=2, dilation=1, stride=2, N=1, **kwargs):
        super().__init__(
            down_conv_nn=up_conv_nn, kernel_size=kernel_size, dilation=dilation, stride=stride, N=N, **kwargs,
        )

    def forward(self, x, skip):
        if skip is not None:
            inp = nn.cat(x, skip)
        else:
            inp = x
        return super().forward(inp)
