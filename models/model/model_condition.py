import torch.nn as nn

from abc import abstractmethod
from abc import ABCMeta
from torch import Tensor
from typing import Any
from typing import Dict
from typing import Type

from tools.bases.register import WithRegister

from models.model.model_dl import IDLModel


# `condition_models` are rather 'independent' models, they can be used independently
condition_models: Dict[str, Type["IConditionModel"]] = {}
# `specialized_condition_models` are specialized for conditioning, they cannot be used independently
specialized_condition_models: Dict[str, Type[nn.Module]] = {}


class IConditionModel(IDLModel, WithRegister, metaclass=ABCMeta):
    d = condition_models

    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m

    @abstractmethod
    def forward(self, cond: Any) -> Tensor:
        pass


class ISpecializedConditionModel(nn.Module, WithRegister, metaclass=ABCMeta):
    d = specialized_condition_models

    @abstractmethod
    def forward(self, cond: Any) -> Tensor:
        pass