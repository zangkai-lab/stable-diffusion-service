import torch
from torch import nn

from typing import Union, Any

from tools.utils.type import tensor_dict_type


def get_torch_device(device: Union[int, str, torch.device]) -> torch.device:
    if isinstance(device, (int, str)):
        device = torch.device(device)
    return device


def is_cpu(device: Union[int, str, torch.device]) -> bool:
    return get_torch_device(device).type == "cpu"


def empty_cuda_cache(device: Union[int, str, torch.device]) -> None:
    device = get_torch_device(device)
    if device.type != "cuda":
        return
    with torch.cuda.device(device):
        torch.cuda.empty_cache()


def get_device(m: nn.Module) -> torch.device:
    params = list(m.parameters())
    return torch.device("cpu") if not params else params[0].device


def to_device(
    batch: tensor_dict_type,
    device: torch.device,
    **kwargs: Any,
) -> tensor_dict_type:
    return {
        k: v.to(device, **kwargs)
        if isinstance(v, torch.Tensor)
        else [
            vv.to(device, **kwargs) if isinstance(vv, torch.Tensor) else vv for vv in v
        ]
        if isinstance(v, list)
        else v
        for k, v in batch.items()
    }