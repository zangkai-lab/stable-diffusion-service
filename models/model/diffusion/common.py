import torch
import torch.nn as nn

from torch.nn import Module
from typing import List, Tuple
from typing import Iterator

from tools.utils.type import tensor_dict_type


class EMA(Module):
    def __init__(
        self,
        decay: float,
        named_parameters: List[Tuple[str, nn.Parameter]],
        *,
        use_num_updates: bool = False,
    ):
        super().__init__()
        self._cache: tensor_dict_type = {}
        self._decay = decay
        self._named_parameters = named_parameters
        for name, param in self.tgt_params:
            self.register_buffer(name, param.data.clone())
        num_updates = torch.tensor(0 if use_num_updates else -1, dtype=torch.int)
        self.register_buffer("num_updates", num_updates)

    @property
    def tgt_params(self) -> Iterator[Tuple[str, nn.Parameter]]:
        return map(
            lambda pair: (pair[0].replace(".", "_"), pair[1]),
            self._named_parameters,
        )

    def forward(self) -> None:
        if not self.training:
            raise ValueError("should not update `EMA` at inference stage")
        if self.num_updates < 0:
            decay = self._decay
        else:
            self.num_updates += 1
            decay = min(self._decay, (1 + self.num_updates) / (10 + self.num_updates))
        for name, param in self.tgt_params:
            ema_attr = getattr(self, name)
            ema = (1.0 - decay) * param.data + decay * ema_attr
            setattr(self, name, ema.clone())

    def train(self, mode: bool = True) -> "EMA":
        super().train(mode)
        if mode:
            for name, param in self.tgt_params:
                cached = self._cache.pop(name, None)
                if cached is not None:
                    param.data = cached
        else:
            for name, param in self.tgt_params:
                if name not in self._cache:
                    self._cache[name] = param.data
                param.data = getattr(self, name).clone()
        return self

    def extra_repr(self) -> str:
        max_str_len = max(len(name) for name, _ in self.tgt_params)
        return "\n".join(
            [f"(0): decay_rate={self._decay}\n(1): Params("]
            + [
                f"  {name:<{max_str_len}s} - Tensor({list(param.shape)})"
                for name, param in self.tgt_params
            ]
            + [")"]
        )