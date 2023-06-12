import torch
import torch.nn as nn

from typing import Any
from typing import Dict
from typing import NamedTuple
from typing import Type
from typing import Optional
from torch import Tensor

from tools.bases.register import WithRegister

from models.model.blocks.convs.basic import Conv2d


discriminator_dict: Dict[str, Type["DiscriminatorBase"]] = {}


class DiscriminatorOutput(NamedTuple):
    output: Tensor
    cond_logits: Optional[Tensor] = None


class DiscriminatorBase(nn.Module, WithRegister["DiscriminatorBase"]):
    d = discriminator_dict

    clf: nn.Module
    net: nn.Module
    cond: Optional[nn.Module]

    def __init__(
        self,
        in_channels: int,
        num_classes: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

    def generate_cond(self, out_channels: int) -> None:
        if self.num_classes is None:
            self.cond = None
        else:
            self.cond = Conv2d(
                out_channels,
                self.num_classes,
                kernel_size=4,
                padding=1,
                stride=1,
            )

    def forward(self, net: torch.Tensor) -> Any:
        feature_map = self.net(net)
        logits = self.clf(feature_map)
        cond_logits = None
        if self.cond is not None:
            cond_logits_map = self.cond(feature_map)
            cond_logits = torch.mean(cond_logits_map, [2, 3])
        return DiscriminatorOutput(logits, cond_logits)