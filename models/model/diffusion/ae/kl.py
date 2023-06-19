import torch
import numpy as np
from torch import Tensor
from typing import Optional, Tuple


class GaussianDistribution:
    key = "distribution"

    def __init__(self, net: Tensor, deterministic: bool = False):
        self.net = net
        self.device = net.device
        self.deterministic = deterministic
        self.mean, log_var = torch.chunk(net, 2, dim=1)
        self.log_var = torch.clamp(log_var, -30.0, 20.0)
        if self.deterministic:
            zeros = torch.zeros_like(self.mean)
            self.var = self.std = zeros
        else:
            self.std = torch.exp(0.5 * self.log_var)
            self.var = torch.exp(self.log_var)

    def sample(self) -> Tensor:
        std = self.std * torch.randn_like(self.mean)
        return self.mean + std

    def kl(self, other: Optional["GaussianDistribution"] = None) -> Tensor:
        if self.deterministic:
            return torch.tensor([0.0], device=self.device)
        if other is None:
            return 0.5 * torch.sum(
                torch.pow(self.mean, 2) + self.var - 1.0 - self.log_var,
                dim=[1, 2, 3],
            )
        return 0.5 * torch.sum(
            torch.pow(self.mean - other.mean, 2) / other.var
            + self.var / other.var
            - 1.0
            - self.log_var
            + other.log_var,
            dim=[1, 2, 3],
        )

    def nll(self, sample: Tensor, dims: Tuple[int, ...] = (1, 2, 3)) -> Tensor:
        if self.deterministic:
            return torch.tensor([0.0], device=self.device)
        return 0.5 * torch.sum(
            np.log(2.0 * np.pi)
            + self.log_var
            + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self) -> Tensor:
        return self.mean