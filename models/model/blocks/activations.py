import torch

import torch.nn as nn
import torch.nn.functional as F

from abc import ABCMeta
from torch import Tensor
from typing import Any
from typing import Dict
from typing import Type

from typing import Optional
from functools import partial
from torch.nn import Module

from tools.bases.register import WithRegister
from models.model.blocks.hijacks import HijackLinear
from models.model.blocks.utils import Lambda


activations: Dict[str, Type["Activation"]] = {}


class Activation(WithRegister["Activation"], Module, metaclass=ABCMeta):
    d = activations

    def __init__(self, **kwargs: Any):
        super().__init__()

    @classmethod
    def make(
        cls,
        name: Optional[str],
        config: Optional[Dict[str, Any]] = None,
    ) -> Module:
        if name is None:
            return nn.Identity()
        if config is None:
            config = {}
        if name.startswith("leaky_relu"):
            splits = name.split("_")
            if len(splits) == 3:
                config["negative_slope"] = float(splits[-1])
            config.setdefault("inplace", True)
            return nn.LeakyReLU(**config)
        if name.lower() == "relu":
            name = "ReLU"
            config.setdefault("inplace", True)
        base = cls.d.get(name, getattr(nn, name, None))
        if base is not None:
            return base(**config)
        func = getattr(torch, name, getattr(F, name, None))
        if func is None:
            raise NotImplementedError(
                "neither pytorch nor custom Activation "
                f"implemented activation '{name}'"
            )
        return Lambda(partial(func, **config), name)


@Activation.register("glu")
class GLU(Activation):
    def __init__(self, *, in_dim: int, bias: bool = True):
        super().__init__()
        self.linear = HijackLinear(in_dim, 2 * in_dim, bias)

    def forward(self, net: Tensor) -> Tensor:
        projection, gate = self.linear(net).chunk(2, dim=1)
        return projection * torch.sigmoid(gate)


@Activation.register("geglu")
class GEGLU(Activation):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = HijackLinear(in_dim, out_dim * 2)

    def forward(self, net: Tensor) -> Tensor:
        net, gate = self.net(net).chunk(2, dim=-1)
        return net * F.gelu(gate)