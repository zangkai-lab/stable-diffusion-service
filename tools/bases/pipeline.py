from abc import abstractmethod
from abc import ABCMeta
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Union
from typing import TypeVar
from typing import Optional

from tools.bases.register import WithRegister
from tools.bases.serializable import ISerializable


TBlock = TypeVar("TBlock", bound="IBlock")
TConfig = TypeVar("TConfig", bound="ISerializableDataClass")
TPipeline = TypeVar("TPipeline", bound="IPipeline")

pipelines: Dict[str, Type["IPipeline"]] = {}
pipeline_blocks: Dict[str, Type["IBlock"]] = {}


def get_req_choices(req: TBlock) -> List[str]:
    return [r.strip() for r in req.__identifier__.split("|")]


def check_requirement(block: "IBlock", previous: Dict[str, "IBlock"]) -> None:
    for req in block.requirements:
        choices = get_req_choices(req)
        if all(c != "none" and c not in previous for c in choices):
            raise ValueError(
                f"'{block.__identifier__}' requires '{req}', "
                "but none is provided in the previous blocks"
            )


class IBlock(WithRegister["IBlock"], metaclass=ABCMeta):
    d = pipeline_blocks

    """
    This property should be injected by the `IPipeline`.
    > In runtime (i.e. executing the `run` method), this property will represent ALL `IBlock`s used in the `IPipeline`.
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


class IPipeline(ISerializable["IPipeline"], metaclass=ABCMeta):
    d = pipelines

    config: TConfig
    blocks: List[TBlock]

    def __init__(self) -> None:
        self.blocks = []

    # abstract

    @classmethod
    @abstractmethod
    def init(cls: Type[TPipeline], config: TConfig) -> TPipeline:
        pass

    @property
    @abstractmethod
    def config_base(self) -> Type[TConfig]:
        pass

    @property
    @abstractmethod
    def block_base(self) -> Type[TBlock]:
        pass

    # inheritance
    # 返回一个包含管道所有信息的字典，可以用于序列化管道状态
    def to_info(self) -> Dict[str, Any]:
        using_serializable_blocks = self.using_serializable_blocks
        return dict(
            blocks=[
                b.to_pack().asdict() if using_serializable_blocks else b.__identifier__
                for b in self.blocks
            ],
            config=self.config.to_pack().asdict(),
        )

    # 从一个信息字典中重建管道状态
    def from_info(self, info: Dict[str, Any]) -> None:
        self.config = self.config_base.from_pack(info["config"])
        block_base = self.block_base
        using_serializable_blocks = self.using_serializable_blocks
        blocks: List[block_base] = []
        for block in info["blocks"]:
            blocks.append(
                block_base.from_pack(block)
                if using_serializable_blocks
                else block_base.make(block, {})
            )
        self.build(*blocks)

    # optional callbacks

    def before_block_build(self, block: TBlock) -> None:
        pass

    def after_block_build(self, block: TBlock) -> None:
        pass

    # api
    # 返回是否所有的块都可以被序列化
    @property
    def using_serializable_blocks(self) -> bool:
        return issubclass(self.block_base, ISerializable)

    @property
    def block_mappings(self) -> Dict[str, TBlock]:
        return {b.__identifier__: b for b in self.blocks}

    def try_get_block(self, block: Union[str, Type[TBlock]]) -> Optional[TBlock]:
        if not isinstance(block, str):
            block = block.__identifier__
        return self.block_mappings.get(block)

    def get_block(self, block: Union[str, Type[TBlock]]) -> TBlock:
        b = self.try_get_block(block)
        if b is None:
            raise ValueError(f"cannot find '{block}' in `previous`")
        return b

    def remove(self, *block_names: str) -> None:
        pop_indices = []
        for i, block in enumerate(self.blocks):
            if block.__identifier__ in block_names:
                pop_indices.append(i)
        for i in pop_indices[::-1]:
            self.blocks.pop(i)

    # 核心方法，用于构建管道
    def build(self, *blocks: TBlock) -> None:
        previous: Dict[str, TBlock] = self.block_mappings
        for block in blocks:
            check_requirement(block, previous)
            block.previous = previous
            self.before_block_build(block)
            block.build(self.config)
            self.after_block_build(block)
            previous[block.__identifier__] = block
            self.blocks.append(block)