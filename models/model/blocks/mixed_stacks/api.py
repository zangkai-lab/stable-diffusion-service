import math
import torch.nn as nn

from torch import Tensor
from typing import Any
from typing import Tuple
from typing import Callable
from typing import Dict
from typing import Optional
from torch.nn import Module

from tools.utils.icopy import shallow_copy_dict

from models.model.blocks.utils import gradient_checkpoint, zero_module, new_seed
from models.model.blocks.hijacks import HijackLinear, HijackConv2d
from models.model.blocks.attentions import CrossAttention
from models.model.blocks.mixed_stacks.channel_mixers import FeedForward


def do_nothing(x: Tensor) -> Tensor:
    return x


def compute_merge(x: Tensor, tome_info: Dict[str, Any]) -> Tuple[Callable, ...]:
    original_h, original_w = tome_info["size"]
    original_tokens = original_h * original_w
    downsample = int(math.ceil(math.sqrt(original_tokens // x.shape[1])))

    if downsample > tome_info["max_downsample"]:
        m, u = do_nothing, do_nothing
    else:
        w = int(math.ceil(original_w / downsample))
        h = int(math.ceil(original_h / downsample))
        r = int(x.shape[1] * tome_info["ratio"])
        m, u = bipartite_soft_matching_random2d(  # type: ignore
            x,
            w,
            h,
            tome_info["sx"],
            tome_info["sy"],
            r,
            tome_info.get("seed", new_seed()),
            not tome_info["use_rand"],
        )

    m_a, u_a = (m, u) if tome_info["merge_attn"] else (do_nothing, do_nothing)
    m_c, u_c = (m, u) if tome_info["merge_crossattn"] else (do_nothing, do_nothing)
    m_m, u_m = (m, u) if tome_info["merge_mlp"] else (do_nothing, do_nothing)

    return m_a, m_c, m_m, u_a, u_c, u_m


class SpatialTransformerBlock(Module):
    def __init__(
        self,
        query_dim: int,
        num_heads: int,
        head_dim: int,
        *,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,
        feedforward_multiplier: float = 4.0,
        feedforward_activation: str = "geglu",
        use_checkpoint: bool = False,
        attn_split_chunk: Optional[int] = None,
        tome_info: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=query_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            attn_split_chunk=attn_split_chunk,
        )
        latent_dim = round(query_dim * feedforward_multiplier)
        self.ff = FeedForward(
            query_dim,
            latent_dim,
            dropout,
            activation=feedforward_activation,
            add_last_dropout=False,
        )
        self.attn2 = CrossAttention(
            query_dim=query_dim,
            context_dim=context_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            attn_split_chunk=attn_split_chunk,
        )
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.norm3 = nn.LayerNorm(query_dim)
        self.use_checkpoint = use_checkpoint
        self.set_tome_info(tome_info)

    # tomesd (https://github.com/dbolya/tomesd)
    def set_tome_info(self, tome_info: Optional[Dict[str, Any]]) -> None:
        if tome_info is None:
            self.tome_info = None
        else:
            self.tome_info = shallow_copy_dict(tome_info)
            self.tome_info.setdefault("ratio", 0.5)
            self.tome_info.setdefault("max_downsample", 1)
            self.tome_info.setdefault("sx", 2)
            self.tome_info.setdefault("sy", 2)
            self.tome_info.setdefault("use_rand", True)
            self.tome_info.setdefault("merge_attn", True)
            self.tome_info.setdefault("merge_crossattn", False)
            self.tome_info.setdefault("merge_mlp", False)

    def forward(self, net: Tensor, context: Optional[Tensor] = None) -> Tensor:
        inputs = (net,) if context is None else (net, context)
        return gradient_checkpoint(
            self._forward,
            inputs=inputs,
            params=self.parameters(),
            enabled=self.use_checkpoint,
        )

    def _forward(self, net: Tensor, context: Optional[Tensor] = None) -> Tensor:
        if self.tome_info is None:
            net = self.attn1(self.norm1(net)) + net
            net = self.attn2(self.norm2(net), context=context) + net
            net = self.ff(self.norm3(net)) + net
        else:
            m_a, m_c, m_m, u_a, u_c, u_m = compute_merge(net, self.tome_info)
            net = u_a(self.attn1(m_a(self.norm1(net)))) + net
            net = u_c(self.attn2(m_c(self.norm2(net)), context=context)) + net
            net = u_m(self.ff(m_m(self.norm3(net)))) + net
        return net


class SpatialTransformer(Module):
    def __init__(
        self,
        in_channels: int,
        num_heads: int,
        head_dim: int,
        *,
        num_layers: int = 1,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,
        use_linear: bool = False,
        use_checkpoint: bool = False,
        attn_split_chunk: Optional[int] = None,
        tome_info: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_channels, 1.0e-6, affine=True)
        self.use_linear = use_linear
        latent_channels = num_heads * head_dim
        if not use_linear:
            self.to_latent = HijackConv2d(in_channels, latent_channels, 1, 1, 0)
        else:
            self.to_latent = HijackLinear(in_channels, latent_channels)
        self.blocks = nn.ModuleList(
            [
                SpatialTransformerBlock(
                    latent_channels,
                    num_heads,
                    head_dim,
                    dropout=dropout,
                    context_dim=context_dim,
                    use_checkpoint=use_checkpoint,
                    attn_split_chunk=attn_split_chunk,
                    tome_info=tome_info,
                )
                for _ in range(num_layers)
            ]
        )
        self.from_latent = zero_module(
            HijackConv2d(latent_channels, in_channels, 1, 1, 0)
            if not use_linear
            else HijackLinear(in_channels, latent_channels)
        )

    def set_tome_info(self, tome_info: Optional[Dict[str, Any]]) -> None:
        for block in self.blocks:
            block.set_tome_info(tome_info)

    def forward(self, net: Tensor, context: Optional[Tensor]) -> Tensor:
        inp = net
        b, c, h, w = net.shape
        net = self.norm(net)
        if not self.use_linear:
            net = self.to_latent(net)
        net = net.permute(0, 2, 3, 1).reshape(b, h * w, c)
        if self.use_linear:
            net = self.to_latent(net)
        for block in self.blocks:
            net = block(net, context=context)
        if self.use_linear:
            net = self.from_latent(net)
        net = net.permute(0, 2, 1).contiguous()
        net = net.view(b, c, h, w)
        if not self.use_linear:
            net = self.from_latent(net)
        return inp + net