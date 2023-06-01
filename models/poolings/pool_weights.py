import torch

from tools.bases.pooling import ILoadableItem
from tools.bases.pooling import ILoadablePool
from tools.utils.type import tensor_dict_type


class Weights(ILoadableItem[tensor_dict_type]):
    def __init__(self, path: str, *, init: bool = False):
        super().__init__(lambda: torch.load(path), init=init)


class WeightsPool(ILoadablePool[tensor_dict_type]):
    def register(self, key: str, path: str) -> None:  # type: ignore
        super().register(key, lambda init: Weights(path, init=init))