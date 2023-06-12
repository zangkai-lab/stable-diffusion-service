from abc import abstractmethod
from abc import ABCMeta
from typing import Any
from typing import Dict
from typing import Type
from tools.bases.register import WithRegister


dl_zoo_model_loaders: Dict[str, Type["IDLZooModelLoader"]] = {}


class IDLZooModelLoader(WithRegister, metaclass=ABCMeta):
    d = dl_zoo_model_loaders

    @abstractmethod
    def permute_kwargs(self, kwargs: Dict[str, Any]) -> None:
        pass