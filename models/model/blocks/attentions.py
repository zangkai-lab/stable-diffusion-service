import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional
from torch.nn import Module

from models.model.blocks.hijacks import HijackLinear
from models.model.blocks.convs.basic import conv_nd
from models.model.blocks.utils import zero_module, gradient_checkpoint


pt2_sdp_attn = getattr(F, "scaled_dot_product_attention", None)

warnings = set()


def _sdp_attn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    training: bool,
    mask: Optional[Tensor] = None,
    dropout: Optional[float] = None,
) -> Tensor:
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    if pt2_sdp_attn is not None:
        dropout = dropout if training else None
        dropout = 0.0 if dropout is None else dropout
        return pt2_sdp_attn(q, k, v, mask, dropout)
    raw_weights = q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1])
    if mask is not None:
        raw_weights.masked_fill_(~mask, float("-inf"))
    weights = F.softmax(raw_weights, dim=-1)
    if training and dropout is not None and 0.0 < dropout < 1.0:
        weights = F.dropout(weights, dropout)
    return weights @ v


def warn_once(message: str, *, key: Optional[str] = None) -> None:
    key = key or message
    if key not in warnings:
        print(message)
        warnings.add(key)


def try_run_xformers_sdp_attn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    training: bool,
    mask: Optional[Tensor] = None,
    p: Optional[float] = None,
) -> Optional[Tensor]:
    try:
        import xformers.ops

        if p is None:
            p = 0.0
        return xformers.ops.memory_efficient_attention(q, k, v, mask, p)
    except Exception as err:
        warn_once(f"failed to run `xformers` sdp attn: {err}")
        return None


def sdp_attn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    training: bool,
    mask: Optional[Tensor] = None,
    dropout: Optional[float] = None,
    split_chunk: Optional[int] = None,
) -> Tensor:
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    try_xformers = try_run_xformers_sdp_attn(q, k, v, training, mask, dropout)
    if try_xformers is not None:
        return try_xformers
    size = q.shape[0]
    if split_chunk is None:
        if mask is not None and len(mask.shape) == 3:
            b = mask.shape[0]
            mask = mask.view(b, -1)
            mask = mask[:, None, :].repeat(size // b, 1, 1)
        return _sdp_attn(q, k, v, training, mask)
    if mask is not None:
        msg = "`mask` is not supported yet when `attn_split_chunk` is enabled"
        raise ValueError(msg)
    tq = q.shape[1]
    net = torch.zeros(size, tq, v.shape[2], dtype=q.dtype, device=q.device)
    for i in range(0, size, split_chunk):
        end = i + split_chunk
        net[i:end] = _sdp_attn(q[i:end], k[i:end], v[i:end], training, dropout=dropout)
    return net


class CrossAttention(Module):
    def __init__(
        self,
        *,
        query_dim: int,
        context_dim: Optional[int] = None,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        attn_split_chunk: Optional[int] = None,
    ):
        super().__init__()
        self.has_context = context_dim is not None
        latent_dim = head_dim * num_heads
        context_dim = context_dim or query_dim

        self.num_heads = num_heads
        self.attn_split_chunk = attn_split_chunk

        self.to_q = HijackLinear(query_dim, latent_dim, bias=False)
        self.to_k = HijackLinear(context_dim, latent_dim, bias=False)
        self.to_v = HijackLinear(context_dim, latent_dim, bias=False)

        self.out_linear = nn.Sequential(
            HijackLinear(latent_dim, query_dim),
            nn.Dropout(dropout),
        )

    def transpose(self, net: Tensor) -> Tensor:
        b, t, d = net.shape
        dim = d // self.num_heads
        # (B, T, D) -> (B, T, head, dim)
        net = net.view(b, t, self.num_heads, dim)
        # (B, T, head, dim) -> (B, head, T, dim)
        net = net.permute(0, 2, 1, 3)
        # (B, head, T, dim) -> (B * head, T, dim)
        net = net.reshape(b * self.num_heads, t, dim)
        return net

    def forward(
        self,
        net: Tensor,
        *,
        context: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        # (B, Tq, Dq)
        b, tq, dq = net.shape

        # (B, Tq, D)
        q = self.to_q(net)
        if context is None:
            context = net
        # (B, Tc, D)
        k = self.to_k(context)
        v = self.to_v(context)

        # (B * head, Tq, dim)
        q = self.transpose(q)
        # (B * head, Tc, dim)
        k = self.transpose(k)
        v = self.transpose(v)

        if mask is not None:
            mask = ~mask
        net = sdp_attn(q, k, v, self.training, mask, split_chunk=self.attn_split_chunk)
        # (B, head, Tq, dim)
        net = net.reshape(b, self.num_heads, tq, dq // self.num_heads)
        # (B, Tq, head, dim)
        net = net.permute(0, 2, 1, 3).contiguous()
        # (B, Tq, D)
        net = net.view(b, tq, dq)
        # (B, Tq, Dq)
        net = self.out_linear(net)
        return net


class MultiHeadSpatialAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        *,
        num_heads: Optional[int] = 1,
        num_head_channels: Optional[int] = None,
        split_qkv_before_heads: bool = False,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        if num_head_channels is None:
            if num_heads is None:
                msg = "either `num_heads` or `num_head_channels` should be provided"
                raise ValueError(msg)
            self.num_heads = num_heads
        else:
            self.num_heads = in_channels // num_head_channels
        self.split_qkv_before_heads = split_qkv_before_heads
        self.use_checkpoint = use_checkpoint
        self.norm = nn.GroupNorm(32, in_channels)
        self.to_qkv = conv_nd(1, in_channels, in_channels * 3, 1)
        self.to_out = zero_module(conv_nd(1, in_channels, in_channels, 1))

    def forward(self, net: Tensor) -> Tensor:
        return gradient_checkpoint(
            self._forward,
            (net,),
            self.parameters(),
            self.use_checkpoint,
        )

    def _forward(self, net: Tensor) -> Tensor:
        b, c, h, w = net.shape
        area = h * w

        inp = net = net.view(b, c, area)
        qkv = self.to_qkv(self.norm(net))
        head_dim = int(c) // self.num_heads
        args = b, c, area, head_dim
        if self.split_qkv_before_heads:
            net = self._split_qkv_before_heads(qkv, *args)
        else:
            net = self._split_qkv_after_heads(qkv, *args)
        net = self.to_out(net)
        return (inp + net).view(b, c, h, w)

    def _split_qkv_before_heads(
        self,
        qkv: Tensor,
        b: int,
        c: int,
        area: int,
        head_dim: int,
    ) -> Tensor:
        q, k, v = qkv.chunk(3, dim=1)

        scale = 1.0 / math.sqrt(math.sqrt(head_dim))
        attn_mat = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(b * self.num_heads, head_dim, area),
            (k * scale).view(b * self.num_heads, head_dim, area),
        )
        attn_prob = F.softmax(attn_mat, dim=-1)
        net = torch.einsum(
            "bts,bcs->bct",
            attn_prob,
            v.contiguous().view(b * self.num_heads, head_dim, area),
        )
        return net.contiguous().view(b, c, area)

    def _split_qkv_after_heads(
        self,
        qkv: Tensor,
        b: int,
        c: int,
        area: int,
        head_dim: int,
    ) -> Tensor:
        qkv = qkv.view(b * self.num_heads, head_dim * 3, area)
        q, k, v = qkv.split(head_dim, dim=1)

        scale = 1.0 / math.sqrt(math.sqrt(head_dim))
        attn_mat = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        attn_prob = F.softmax(attn_mat, dim=-1)
        net = torch.einsum("bts,bcs->bct", attn_prob, v)
        return net.contiguous().view(b, c, area)