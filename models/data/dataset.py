import numpy as np

from abc import ABC, abstractmethod
from typing import Dict, Any, Union, List


class IDataset(ABC):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, item: Union[int, List[int], np.ndarray]) -> Dict[str, Any]:
        pass