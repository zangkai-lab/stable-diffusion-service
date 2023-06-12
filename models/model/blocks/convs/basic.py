import math
import torch
import torch.nn as nn

from typing import Any, Optional, Union
from torch.nn import Module
from torch import Tensor
import torch.nn.functional as F

from models.model.blocks.hijacks import HijackConv1d
from models.model.blocks.hijacks import HijackConv2d
from models.model.blocks.hijacks import HijackConv3d


def conv_nd(n: int, *args: Any, **kwargs: Any) -> Module:
    if n == 1:
        return HijackConv1d(*args, **kwargs)
    elif n == 2:
        return HijackConv2d(*args, **kwargs)
    elif n == 3:
        return HijackConv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {n}")


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        groups: int = 1,
        stride: int = 1,
        dilation: int = 1,
        padding: Any = "same",
        transform_kernel: bool = False,
        bias: bool = True,
        demodulate: bool = False,
        weight_scale: Optional[float] = None,
        gain: float = math.sqrt(2.0),
    ):
        super().__init__()
        self.in_c, self.out_c = in_channels, out_channels
        self.kernel_size = kernel_size
        self.reflection_pad = None
        if padding == "same":
            padding = kernel_size // 2
        elif isinstance(padding, str) and padding.startswith("reflection"):
            reflection_padding: Any
            if padding == "reflection":
                reflection_padding = kernel_size // 2
            else:
                reflection_padding = int(padding[len("reflection") :])
            padding = 0
            if transform_kernel:
                reflection_padding = [reflection_padding] * 4
                reflection_padding[0] += 1
                reflection_padding[2] += 1
            self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.groups, self.stride = groups, stride
        self.dilation, self.padding = dilation, padding
        self.transform_kernel = transform_kernel
        self.weight = nn.Parameter(
            torch.empty(
                out_channels,
                in_channels // groups,
                kernel_size,
                kernel_size,
            )
        )
        if not bias:
            self.bias = None
        else:
            self.bias = nn.Parameter(torch.empty(out_channels))
        self.demodulate = demodulate
        self.weight_scale = weight_scale
        # initialize
        with torch.no_grad():
            nn.init.xavier_normal_(self.weight.data, gain / math.sqrt(2.0))
            if self.bias is not None:
                self.bias.zero_()

    def _same_padding(self, size: int) -> int:
        stride = self.stride
        dilation = self.dilation
        return ((size - 1) * (stride - 1) + dilation * (self.kernel_size - 1)) // 2

    def forward(
        self,
        net: Tensor,
        style: Optional[Tensor] = None,
        *,
        transpose: bool = False,
    ) -> Tensor:
        b, c, *hw = net.shape
        # padding
        padding = self.padding
        if self.padding == "same":
            padding = tuple(map(self._same_padding, hw))
        if self.reflection_pad is not None:
            net = self.reflection_pad(net)
        # transform kernel
        w: Union[nn.Parameter, Tensor] = self.weight
        if self.transform_kernel:
            w = F.pad(w, [1, 1, 1, 1], mode="constant")
            w = (
                w[:, :, 1:, 1:]
                + w[:, :, :-1, 1:]
                + w[:, :, 1:, :-1]
                + w[:, :, :-1, :-1]
            ) * 0.25
        # ordinary convolution
        if style is None:
            bias = self.bias
            groups = self.groups
        # 'stylized' convolution, used in StyleGAN
        else:
            suffix = "when `style` is provided"
            if self.bias is not None:
                raise ValueError(f"`bias` should not be used {suffix}")
            if self.groups != 1:
                raise ValueError(f"`groups` should be 1 {suffix}")
            if self.reflection_pad is not None:
                raise ValueError(
                    f"`reflection_pad` should not be used {suffix}, "
                    "maybe you want to use `same` padding?"
                )
            w = w[None, ...] * style[..., None, :, None, None]
            # prepare for group convolution
            bias = None
            groups = b
            net = net.view(1, -1, *hw)  # 1, b*in, h, w
            w = w.view(b * self.out_c, *w.shape[2:])  # b*out, in, wh, ww
        if self.demodulate:
            w = w * torch.rsqrt(w.pow(2).sum([-3, -2, -1], keepdim=True) + 1e-8)
        if self.weight_scale is not None:
            w = w * self.weight_scale
        # conv core
        if not transpose:
            fn = F.conv2d
        else:
            fn = F.conv_transpose2d
            if groups == 1:
                w = w.transpose(0, 1)
            else:
                oc, ic, kh, kw = w.shape
                w = w.reshape(groups, oc // groups, ic, kh, kw)
                w = w.transpose(1, 2)
                w = w.reshape(groups * ic, oc // groups, kh, kw)
        net = fn(
            net,
            w,
            bias,
            stride=self.stride,
            padding=padding,
            dilation=self.dilation,
            groups=groups,
        )
        if style is None:
            return net
        return net.view(b, -1, *net.shape[2:])

    def extra_repr(self) -> str:
        return (
            f"{self.in_c}, {self.out_c}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding}, dilation={self.dilation}, "
            f"bias={self.bias is not None}, demodulate={self.demodulate}"
        )