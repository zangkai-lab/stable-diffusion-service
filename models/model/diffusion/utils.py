import torch

from torch import Tensor
from typing import Union, Optional

from tools.utils.type import tensor_dict_type


cond_type = Union[Tensor, tensor_dict_type]
ADM_KEY = "labels"
ADM_TYPE = "adm"
CONCAT_KEY = "concat"
CONCAT_TYPE = "concat"
HYBRID_TYPE = "hybrid"
CROSS_ATTN_KEY = "context"
CROSS_ATTN_TYPE = "cross_attn"
CONTROL_HINT_KEY = "hint"
CONTROL_HINT_START_KEY = "hint_start"


def extract_to(array: Tensor, indices: Tensor, num_dim: int) -> Tensor:
    b = indices.shape[0]
    out = array.gather(-1, indices).contiguous()
    return out.view(b, *([1] * (num_dim - 1)))


def get_timesteps(t: int, num: int, device: torch.device) -> Tensor:
    return torch.full((num,), t, device=device, dtype=torch.long)


# 球面线性插值
def slerp(
    x1: torch.Tensor,
    x2: torch.Tensor,
    r1: Union[float, torch.Tensor],
    r2: Optional[Union[float, torch.Tensor]] = None,
    *,
    dot_threshold: float = 0.9995,
) -> torch.Tensor:
    if r2 is None:
        r2 = 1.0 - r1
    b, *shape = x1.shape
    x1 = x1.view(b, -1)
    x2 = x2.view(b, -1)
    low_norm = x1 / torch.norm(x1, dim=1, keepdim=True)
    high_norm = x2 / torch.norm(x2, dim=1, keepdim=True)
    dot = (low_norm * high_norm).sum(1)
    overflow_mask = dot > dot_threshold
    out = torch.zeros_like(x1)
    out[overflow_mask] = r1 * x1 + r2 * x2
    normal_mask = ~overflow_mask
    omega = torch.acos(dot[normal_mask])
    so = torch.sin(omega)
    x1_part = (torch.sin(r1 * omega) / so).unsqueeze(1) * x1
    x2_part = (torch.sin(r2 * omega) / so).unsqueeze(1) * x2
    out[normal_mask] = x1_part + x2_part
    return out.view(b, *shape)