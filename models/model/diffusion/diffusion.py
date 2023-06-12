import torch.nn as nn

from torch import Tensor
from typing import Callable, Optional, Protocol

from models.model.diffusion.utils import cond_type


class Denoise(Protocol):
    def __call__(
        self,
        image: Tensor,
        timesteps: Tensor,
        cond: Optional[cond_type],
        step: int,
        total_step: int,
    ) -> Tensor:
        pass


class IDiffusion:
    _get_cond: Callable
    predict_eps_from_z_and_v: Callable
    predict_start_from_z_and_v: Callable

    denoise: Denoise
    q_sampler: "DDPMQSampler"
    condition_model: Optional[nn.Module]
    first_stage: nn.Module

    t: int
    parameterization: str
    posterior_coef1: Tensor
    posterior_coef2: Tensor
    posterior_log_variance_clipped: Tensor

    betas: Tensor
    alphas_cumprod: Tensor
    alphas_cumprod_prev: Tensor