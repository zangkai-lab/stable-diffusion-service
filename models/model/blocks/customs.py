import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Any
from typing import Dict
from typing import Callable
from typing import Union
from typing import List
from typing import Optional
from torch.nn import Module
from functools import partial

from models.model.blocks.hooks import IBasicHook


class Linear(Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        bias: bool = True,
        pruner_config: Optional[Dict[str, Any]] = None,
        init_method: Optional[str] = None,
        rank: Optional[int] = None,
        rank_ratio: Optional[float] = None,
        hook: Optional[IBasicHook] = None,
    ):
        super().__init__()
        original_rank = min(in_dim, out_dim)
        if rank is None:
            if rank_ratio is not None:
                rank = round(original_rank * rank_ratio)
        if rank is not None and rank >= original_rank:
            print(
                f"specified rank ({rank}) should be smaller than "
                f"the original rank ({original_rank})"
            )
        if rank is None:
            self.w1 = self.w2 = self.b = None
            self.linear = nn.Linear(in_dim, out_dim, bias)
        else:
            self.w1 = nn.Parameter(torch.zeros(rank, in_dim))
            self.w2 = nn.Parameter(torch.zeros(out_dim, rank))
            self.b = nn.Parameter(torch.zeros(1, out_dim)) if bias else None
            self.linear = None
        if pruner_config is None:
            self.pruner = self.pruner1 = self.pruner2 = None
        else:
            if rank is None:
                self.pruner1 = self.pruner2 = None
                self.pruner = Pruner(pruner_config, [out_dim, in_dim])
            else:
                self.pruner1 = Pruner(pruner_config, [rank, in_dim])
                self.pruner2 = Pruner(pruner_config, [out_dim, rank])
                self.pruner = None
        # initialize
        init_method = init_method or "xavier_normal"
        init_fn = getattr(nn.init, f"{init_method}_", nn.init.xavier_normal_)
        self.init_weights_with(lambda t: init_fn(t, 1.0 / math.sqrt(2.0)))
        # hook
        self.hook = hook

    @property
    def weight(self) -> Tensor:
        if self.linear is not None:
            return self.linear.weight
        return torch.matmul(self.w2, self.w1)

    @property
    def bias(self) -> Optional[Tensor]:
        return self.b if self.linear is None else self.linear.bias

    def forward(self, net: Tensor) -> Tensor:
        inp = net
        if self.hook is not None:
            inp = self.hook.before_forward(inp)
        if self.linear is not None:
            weight = self.linear.weight
            if self.pruner is not None:
                weight = self.pruner(weight)
            net = F.linear(net, weight, self.linear.bias)
        else:
            w1, w2 = self.w1, self.w2
            if self.pruner1 is not None:
                w1 = self.pruner1(w1)
            if self.pruner2 is not None:
                w2 = self.pruner2(w2)
            net = F.linear(net, w1)
            net = F.linear(net, w2, self.b)
        if self.hook is not None:
            net = self.hook.after_forward(inp, net)
        return net

    def init_weights_with(self, w_init_fn: Callable[[Tensor], None]) -> None:
        with torch.no_grad():
            if self.linear is not None:
                w_init_fn(self.linear.weight.data)
            elif self.w1 is not None and self.w2 is not None:
                w_init_fn(self.w1.data)
                w_init_fn(self.w2.data)
            if self.linear is None:
                if self.b is not None:
                    self.b.data.zero_()
            else:
                if self.linear.bias is not None:
                    self.linear.bias.data.zero_()


class Pruner(Module):
    def __init__(self, config: Dict[str, Any], w_shape: Optional[List[int]] = None):
        super().__init__()
        self.eps: Tensor
        self.exp: Tensor
        self.alpha: Union[Tensor, nn.Parameter]
        self.beta: Union[Tensor, nn.Parameter]
        self.gamma: Union[Tensor, nn.Parameter]
        self.max_ratio: Union[Tensor, nn.Parameter]
        tensor = partial(torch.tensor, dtype=torch.float32)
        self.method = config.setdefault("method", "auto_prune")
        if self.method == "surgery":
            if w_shape is None:
                msg = "`w_shape` of `Pruner` should be provided when `surgery` is used"
                raise ValueError(msg)
            self.register_buffer("mask", torch.ones(*w_shape, dtype=torch.float32))
            self.register_buffer("alpha", tensor([config.setdefault("alpha", 1.0)]))
            self.register_buffer("beta", tensor([config.setdefault("beta", 4.0)]))
            self.register_buffer("gamma", tensor([config.setdefault("gamma", 1e-4)]))
            self.register_buffer("eps", tensor([config.setdefault("eps", 1e-12)]))
            keys = ["alpha", "beta", "gamma", "eps"]
        elif self.method == "simplified":
            self.register_buffer("alpha", tensor([config.setdefault("alpha", 0.01)]))
            self.register_buffer("beta", tensor([config.setdefault("beta", 1.0)]))
            self.register_buffer(
                "max_ratio", tensor([config.setdefault("max_ratio", 1.0)])
            )
            self.register_buffer("exp", tensor([config.setdefault("exp", 0.5)]))
            keys = ["alpha", "beta", "max_ratio", "exp"]
        else:
            self.register_buffer(
                "alpha",
                tensor(
                    [
                        config.setdefault(
                            "alpha", 1e-4 if self.method == "hard_prune" else 1e-2
                        )
                    ]
                ),
            )
            self.register_buffer("beta", tensor([config.setdefault("beta", 1.0)]))
            self.register_buffer("gamma", tensor([config.setdefault("gamma", 1.0)]))
            self.register_buffer(
                "max_ratio", tensor([config.setdefault("max_ratio", 1.0)])
            )
            if not all(
                scalar > 0
                for scalar in [self.alpha, self.beta, self.gamma, self.max_ratio]
            ):
                raise ValueError("parameters should greater than 0. in pruner")
            self.register_buffer("eps", tensor([config.setdefault("eps", 1e-12)]))
            if self.method == "auto_prune":
                for attr in ["alpha", "beta", "gamma", "max_ratio"]:
                    setattr(self, attr, torch.log(torch.exp(getattr(self, attr)) - 1))
                self.alpha, self.beta, self.gamma, self.max_ratio = map(
                    lambda param: nn.Parameter(param),
                    [self.alpha, self.beta, self.gamma, self.max_ratio],
                )
            keys = ["alpha", "beta", "gamma", "max_ratio", "eps"]
        self._repr_keys = keys

    def forward(self, w: Tensor) -> Tensor:
        w_abs = torch.abs(w)
        if self.method == "surgery":
            mu, std = torch.mean(w_abs), torch.std(w_abs)
            zeros_mask = self.mask == 0.0
            ones_mask = self.mask == 1.0
            to_zeros_mask = ones_mask & (w_abs <= 0.9 * (mu - self.beta * std))  # type: ignore
            to_ones_mask = zeros_mask & (w_abs >= 1.1 * (mu + self.beta * std))  # type: ignore
            self.mask.masked_fill(to_zeros_mask, 0.0)  # type: ignore
            self.mask.masked_fill(to_ones_mask, 1.0)  # type: ignore
            mask = self.mask
            del mu, std, ones_mask, zeros_mask, to_zeros_mask, to_ones_mask
        else:
            if self.method != "auto_prune":
                alpha, beta, ratio = self.alpha, self.beta, self.max_ratio
            else:
                alpha, beta, ratio = map(
                    F.softplus,
                    [self.alpha, self.beta, self.max_ratio],
                )
            if self.method == "simplified":
                log_w = torch.min(ratio, beta * torch.pow(w_abs, self.exp))
            else:
                w_abs_mean = torch.mean(w_abs)
                if self.method != "auto_prune":
                    gamma = self.gamma
                else:
                    gamma = F.softplus(self.gamma)
                log_w = torch.log(torch.max(self.eps, w_abs / (w_abs_mean * gamma)))
                log_w = torch.min(ratio, beta * log_w)
                del w_abs_mean
            mask = torch.max(alpha / beta * log_w, log_w)
            del log_w
        w = w * mask
        del w_abs, mask
        return w

    def extra_repr(self) -> str:
        if self.method == "auto_prune":
            return f"method='{self.method}'"
        max_str_len = max(map(len, self._repr_keys))
        return "\n".join(
            [f"(0): method={self.method}\n(1): Settings("]
            + [
                f"  {key:<{max_str_len}s} - {getattr(self, key).item()}"
                for key in self._repr_keys
            ]
            + [")"]
        )