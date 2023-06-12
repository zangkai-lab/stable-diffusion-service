import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

from torch.nn import Module
from torch import Tensor
from typing import Optional, Any, Callable

from models.model.blocks.convs.basic import conv_nd


def avg_pool_nd(n: int, *args: Any, **kwargs: Any) -> Module:
    if n == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif n == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif n == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {n}")


def zero_module(module: Module) -> Module:
    for p in module.parameters():
        p.detach().zero_()
    return module


class ResDownsample(Module):
    def __init__(
        self,
        in_channels: int,
        use_conv: bool,
        *,
        signal_dim: int = 2,
        out_channels: Optional[int] = None,
        padding: int = 1,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        stride = 2 if signal_dim != 3 else (1, 2, 2)
        if not use_conv:
            if in_channels != out_channels:
                raise ValueError(
                    "`in_channels` should be equal to `out_channels` "
                    "when `use_conv` is set to False"
                )
            self.net = avg_pool_nd(signal_dim, kernel_size=stride, stride=stride)
        else:
            self.net = conv_nd(
                signal_dim,
                in_channels,
                out_channels,
                3,
                stride=stride,
                padding=padding,
            )

    def forward(self, net: Tensor) -> Tensor:
        return self.net(net)


class ResUpsample(Module):
    def __init__(
        self,
        in_channels: int,
        use_conv: bool,
        *,
        signal_dim: int = 2,
        out_channels: Optional[int] = None,
        padding: int = 1,
    ):
        super().__init__()
        self.signal_dim = signal_dim
        if not use_conv:
            self.conv = None
        else:
            self.conv = conv_nd(
                signal_dim,
                in_channels,
                out_channels,
                3,
                padding=padding,
            )

    def forward(self, net: Tensor) -> Tensor:
        if self.signal_dim == 3:
            _, _, c, h, w = net.shape
            net = F.interpolate(net, (c, h * 2, w * 2), mode="nearest")
        else:
            net = F.interpolate(net, scale_factor=2, mode="nearest")
        if self.conv is not None:
            net = self.conv(net)
        return net


def gradient_checkpoint(func: Callable, inputs: Any, params: Any, enabled: bool) -> Any:
    if not enabled:
        return func(*inputs)
    args = tuple(inputs) + tuple(params)
    return GradientCheckpointFunction.apply(func, len(inputs), *args)


class GradientCheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        run_function, length, *args = args  # type: ignore
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            grad_outputs,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def safe_clip_(net: Tensor) -> None:
    finfo = torch.finfo(net.dtype)
    net.clamp_(finfo.min, finfo.max)


class Lambda(Module):
    def __init__(self, fn: Callable, name: Optional[str] = None):
        super().__init__()
        self.name = name
        self.fn = fn

    def extra_repr(self) -> str:
        return "" if self.name is None else self.name

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)


min_seed_value = np.iinfo(np.uint32).min
max_seed_value = np.iinfo(np.uint32).max


def new_seed() -> int:
    return random.randint(min_seed_value, max_seed_value)