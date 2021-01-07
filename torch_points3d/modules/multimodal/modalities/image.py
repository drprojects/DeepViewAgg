import torch.nn as nn
import sys
from torch_points3d.utils.config import is_list
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
    """

    # TODO: look into Conv2d(padding_mode='circular') OR torchvision.ops.deform_conv to wrap spherical images and improve border features
    # TODO: padding circular affects coordinates, beware of mappings, beware of mappings validity
    # TODO: mask and crop affects coordinates, beware of mappings validity
    # TODO: extend to EquiConv https: // github.com / palver7 / EquiConvPytorch
    # TODO: optional maxpool with SegNet structure: https: // github.com / say4n / pytorch - segnet / blob / master / src / model.py

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

    def __init__(self, input_nc, output_nc, convolution, reduction=4, **kwargs):
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


SPECIAL_NAMES = ["block_names"]


class UNet(nn.Module):
    """Generic UNet module for images.

    Create the Unet from a dictionary of compact options.

    For each part of the architecture, the blocks are implicitly pre-selected:
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
        self.down_modules = nn.ModuleList()
        self.inner_modules = nn.ModuleList()
        self.up_modules = nn.ModuleList()

        # Down modules
        for i in range(len(opt.down_conv.down_conv_nn)):
            down_module = self._build_module(opt.down_conv, i, "DOWN")
            self.down_modules.append(down_module)

        # Innermost module
        contains_global = hasattr(opt, "innermost") and opt.innermost is not None
        if contains_global:
            inners = self._build_module(opt.innermost, 0, "INNER")
            self.inner_modules.append(inners)
        else:
            self.inner_modules.append(Identity())

        # Up modules
        for i in range(len(opt.up_conv.up_conv_nn)):
            up_module = self._build_module(opt.up_conv, i, "UP")
            self.up_modules.append(up_module)


    def _fetch_arguments_from_list(self, opt, index):
        """Fetch the arguments for a single convolution from multiple lists
        of arguments - for models specified in the compact format.
        """
        args = {}
        for o, v in opt.items():
            name = str(o)
            if is_list(v) and len(getattr(opt, o)) > 0:
                if name[-1] == "s" and name not in SPECIAL_NAMES:
                    name = name[:-1]
                v_index = v[index]
                if is_list(v_index):
                    v_index = list(v_index)
                args[name] = v_index
            else:
                if is_list(v):
                    v = list(v)
                args[name] = v
        return args

    def _build_module(self, opt, index, flow):
        """Builds a convolution (up, down or inner) block.

        Arguments:
            conv_opt - model config subset describing the convolutional block
            index - layer index in sequential order (as they come in the config)
            flow - UP, DOWN or INNER
        """
        if flow.lower() == 'DOWN'.lower():
            module_cls = ResNetDown
        elif flow.lower() == 'INNER'.lower():
            module_cls = BottleneckBlock
        elif flow.lower() == 'UP'.lower():
            module_cls = ResNetUp
        else:
            raise NotImplementedError
        args = self._fetch_arguments_from_list(opt, index)
        return module_cls(**args)

    def forward(self, x, **kwargs):
        """ This method does a forward on the Unet assuming symmetrical skip connections

        Parameters
        ----------
        x: torch.Tensor of images [BxCxHxW]
        """
        stack_down = []
        for i in range(len(self.down_modules) - 1):
            x = self.down_modules[i](x)
            stack_down.append(x)
        x = self.down_modules[-1](x)

        if not isinstance(self.inner_modules[0], Identity):
            stack_down.append(x)
            x = self.inner_modules[0](x)

        for i in range(len(self.up_modules)):
            x = self.up_modules[i]((x, stack_down.pop()))

        return x
