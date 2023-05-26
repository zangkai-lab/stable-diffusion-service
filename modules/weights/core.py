import torch

from typing import Dict, Union, Any

from modules.apis import ILoadableItem
from modules.apis import ILoadablePool


tensor_dict_type = Dict[str, Union[torch.Tensor, Any]]


class Weights(ILoadableItem[tensor_dict_type]):
    def __init__(self, path: str, *, init: bool = False):
        super().__init__(lambda: torch.load(path), init=init)


class WeightsPool(ILoadablePool[tensor_dict_type]):
    def register(self, key: str, path: str) -> None:  # type: ignore
        super().register(key, lambda init: Weights(path, init=init))