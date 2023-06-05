import math
import numpy as np

from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Any

from models.data.dataset import IDataset

from tools.utils.type import np_dict_type
from tools.handler.context import context_error_handler

TDataLoaders = Tuple["IDataLoader", Optional["IDataLoader"]]


class IDataLoader(ABC):
    dataset: IDataset
    batch_size: int

    def __init__(self, *, sample_weights: Optional[np.ndarray] = None):
        self.sample_weights = sample_weights

    @abstractmethod
    def __iter__(self) -> "IDataLoader":
        pass

    @abstractmethod
    def __next__(self) -> np_dict_type:
        pass

    @abstractmethod
    def disable_shuffle(self) -> None:
        pass

    @abstractmethod
    def recover_shuffle(self) -> None:
        pass

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)

    def copy(self) -> "IDataLoader":
        return deepcopy(self)

    def temporarily_disable_shuffle(self) -> context_error_handler:
        class _(context_error_handler):
            def __init__(self, loader: IDataLoader):
                self.loader = loader

            def __enter__(self) -> None:
                self.loader.disable_shuffle()

            def _normal_exit(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                self.loader.recover_shuffle()

        return _(self)

    def get_full_batch(self) -> np_dict_type:
        batch_size = self.batch_size
        self.batch_size = len(self.dataset)
        full_batch = next(iter(self))
        self.batch_size = batch_size
        return full_batch

