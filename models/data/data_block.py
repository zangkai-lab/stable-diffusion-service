import numpy as np

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List

from tools.bases.serializable import ISerializable
from tools.bases.pipeline import IBlock
from tools.utils.ddp import is_local_rank_0

from models.data.data_bundle import DataBundle
from tools.mixin.from_info import PureFromInfoMixin


class IDataBlock(PureFromInfoMixin, IBlock, ISerializable, metaclass=ABCMeta):
    config: "DataProcessorConfig"
    previous: Dict[str, "IDataBlock"]

    def __init__(self, **kwargs: Any) -> None:
        not_exists_tag = "$$NOT_EXISTS$$"
        for field in self.fields:
            value = kwargs.get(field, not_exists_tag)
            if value == not_exists_tag:
                raise ValueError(
                    f"Argument '{field}' needs to be provided "
                    f"for `{self.__class__.__name__}`."
                )
            setattr(self, field, value)

    # inherit

    def build(self, config: "DataProcessorConfig") -> None:
        self.config = config
        configs = (config.block_configs or {}).setdefault(self.__identifier__, {})
        for field in self.fields:
            setattr(self, field, configs.setdefault(field, getattr(self, field)))

    def to_info(self) -> Dict[str, Any]:
        return {field: getattr(self, field) for field in self.fields}

    # abstract

    @property
    @abstractmethod
    def fields(self) -> List[str]:
        pass

    @abstractmethod
    def transform(self, bundle: DataBundle, for_inference: bool) -> DataBundle:
        """
        This method should not utilize `config`!

        Changes can happen inplace.
        """

    @abstractmethod
    def fit_transform(self, bundle: DataBundle) -> DataBundle:
        """
        This method should prepare necessary info, which might be used
        in the `to_info` method.

        If any necessary info comes from `config`, this method should extract
        them and assign them to the corresponding properties.

        This method will NOT be called in a loading procedure, and the
        necessary info should be loaded in the `from_info` method.

        This method will always assume `for_inference=False`.

        Changes can happen inplace.
        """

    # optional callbacks

    # changes can happen inplace
    def postprocess_item(self, item: Any) -> Any:
        return item

    # changes can happen inplace
    def recover_labels(self, y: np.ndarray) -> np.ndarray:
        return y

    # api

    @property
    def is_local_rank_0(self) -> bool:
        return is_local_rank_0()