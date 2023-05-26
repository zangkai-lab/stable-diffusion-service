import torch
from torch import nn

from typing import Union


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
