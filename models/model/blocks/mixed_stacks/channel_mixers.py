import torch.nn as nn

from torch.nn import Module
from torch import Tensor
from typing import Dict, Type
from abc import abstractmethod

from tools.bases.register import WithRegister
from models.model.blocks.hijacks import HijackCustomLinear
from models.model.blocks.activations import Activation, GEGLU


channel_mixers: Dict[str, Type["ChannelMixerBase"]] = {}


class ChannelMixerBase(Module, WithRegister["ChannelMixerBase"]):
    d = channel_mixers

    def __init__(self, in_dim: int, latent_dim: int, dropout: float):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.dropout = dropout

    @property
    @abstractmethod
    def need_2d(self) -> bool:
        pass

    @abstractmethod
    def forward(self, net: Tensor) -> Tensor:
        pass


@ChannelMixerBase.register("ff")
class FeedForward(ChannelMixerBase):
    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        dropout: float,
        activation: str = "GELU",
        add_last_dropout: bool = True,
    ):
        super().__init__(in_dim, latent_dim, dropout)
        if activation == "geglu":
            blocks = [GEGLU(in_dim, latent_dim)]
        else:
            blocks = [
                HijackCustomLinear(in_dim, latent_dim),
                Activation.make(activation),
            ]
        blocks += [nn.Dropout(dropout), HijackCustomLinear(latent_dim, in_dim)]
        if add_last_dropout:
            blocks.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*blocks)

    @property
    def need_2d(self) -> bool:
        return False

    def forward(self, net: Tensor) -> Tensor:
        return self.net(net)