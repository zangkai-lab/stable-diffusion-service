from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Type, Union, TypeVar
from .withRegister import WithRegister

TBlock = TypeVar("TBlock", bound="IBlock")
TConfig = TypeVar("TConfig", bound="ISerializableDataClass")
pipeline_blocks: Dict[str, Type["IBlock"]] = {}


# 可以管理所有IBlock子类
class IBlock(WithRegister["IBlock"], metaclass=ABCMeta):
    d = pipeline_blocks

    """
    This property should be injected by the `IPipeline`.
    > In runtime (i.e. executing the `run` method), this property will represent ALL `IBlock`s used in the `IPipeline`.
    1. 它的键是 IBlock 的名字（__identifier__），值是 IBlock 的实例
    2. 这个属性可以被 IPipeline 类注入，用于在运行时表示 IPipeline 中使用的所有 IBlock
    """
    previous: Dict[str, TBlock]

    @abstractmethod
    def build(self, config: TConfig) -> None:
        """This method can modify the `config` inplace, which will affect the following blocks"""

    @property
    def requirements(self) -> List[Type[TBlock]]:
        return []

    def try_get_previous(self, block: Union[str, Type[TBlock]]) -> Optional[TBlock]:
        if not isinstance(block, str):
            block = block.__identifier__
        return self.previous.get(block)

    def get_previous(self, block: Union[str, Type[TBlock]]) -> TBlock:
        b = self.try_get_previous(block)
        if b is None:
            raise ValueError(f"cannot find '{block}' in `previous`")
        return b