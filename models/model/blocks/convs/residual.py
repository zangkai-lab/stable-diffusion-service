import torch
import torch.nn as nn

from torch.nn import Module
from torch import Tensor
from typing import Optional, Any

from models.model.blocks.hijacks import HijackLinear, HijackConv2d
from models.model.blocks.convs.basic import conv_nd
from models.model.blocks.utils import ResDownsample
from models.model.blocks.utils import ResUpsample
from models.model.blocks.utils import zero_module
from models.model.blocks.utils import gradient_checkpoint
from models.model.blocks.utils import safe_clip_


class ResidualBlockWithTimeEmbedding(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        *,
        signal_dim: int = 2,
        dropout: float = 0.0,
        norm_eps: float = 1.0e-6,
        use_conv_shortcut: bool = False,
        integrate_upsample: bool = False,
        integrate_downsample: bool = False,
        time_embedding_channels: int = 512,
        use_scale_shift_norm: bool = False,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = out_channels or in_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = use_conv_shortcut
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_checkpoint = use_checkpoint

        self.resample = integrate_upsample or integrate_downsample
        if not self.resample:
            self.inp_resample = self.net_resample = None
        elif integrate_upsample:
            self.inp_resample = ResUpsample(in_channels, False, signal_dim=signal_dim)
            self.net_resample = ResUpsample(in_channels, False, signal_dim=signal_dim)
        else:
            self.inp_resample = ResDownsample(in_channels, False, signal_dim=signal_dim)
            self.net_resample = ResDownsample(in_channels, False, signal_dim=signal_dim)

        make_norm = lambda c: nn.GroupNorm(num_groups=32, num_channels=c, eps=norm_eps)

        self.activation = nn.SiLU()
        self.norm1 = make_norm(in_channels)
        self.conv1 = conv_nd(signal_dim, in_channels, out_channels, 3, 1, 1)
        if time_embedding_channels > 0:
            if use_scale_shift_norm:
                t_out_channels = 2 * out_channels
            else:
                t_out_channels = out_channels
            self.time_embedding = HijackLinear(time_embedding_channels, t_out_channels)
        self.norm2 = make_norm(out_channels)
        self.dropout = nn.Dropout(dropout)
        conv2 = conv_nd(signal_dim, out_channels, out_channels, 3, 1, 1)
        self.conv2 = zero_module(conv2)
        if in_channels != out_channels:
            if use_conv_shortcut:
                self.shortcut = HijackConv2d(in_channels, out_channels, 3, 1, 1)
            else:
                self.shortcut = HijackConv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, net: Tensor, time_net: Optional[Tensor] = None) -> Tensor:
        inputs = (net,) if time_net is None else (net, time_net)
        return gradient_checkpoint(
            self._forward,
            inputs=inputs,
            params=self.parameters(),
            enabled=self.use_checkpoint,
        )

    def _forward(self, net: Tensor, time_net: Optional[Tensor] = None) -> Tensor:
        inp = net
        net = self.norm1(net)
        net = self.activation(net)
        if self.inp_resample is None or self.net_resample is None:
            net = self.conv1(net)
        else:
            inp = self.inp_resample(inp)
            net = self.net_resample(net)
            net = self.conv1(net)
        if self.in_channels != self.out_channels:
            inp = self.shortcut(inp)

        if time_net is not None:
            time_net = self.activation(time_net)
            time_net = self.time_embedding(time_net)
            while len(time_net.shape) < len(net.shape):
                time_net = time_net[..., None]
            if self.use_scale_shift_norm:
                scale, shift = torch.chunk(time_net, 2, dim=1)
                net = self.norm2(net) * (1.0 + scale) + shift
                net = self.activation(net)
                net = self.dropout(net)
                net = self.conv2(net)
                return inp + net
            net = net + time_net

        net = self.norm2(net)
        net = self.activation(net)
        net = self.dropout(net)
        net = self.conv2(net)

        net = inp + net
        # 对于输出结果进行截断
        safe_clip_(net)

        return net