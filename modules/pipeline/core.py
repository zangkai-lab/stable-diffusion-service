from typing import Any, Dict, List, Optional, Type, TypeVar, Union
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

from .utils import shallow_copy_dict
from .mixins import IBlock
from .mixins import IData
from .mixins import ISerializable
from .mixins import DLConfig
from .utils import safe_execute
from .utils import get_ddp_info


TBlock = TypeVar("TBlock", bound="IBlock")
TConfig = TypeVar("TConfig", bound="ISerializableDataClass")
TPipeline = TypeVar("TPipeline", bound="IPipeline")

pipelines: Dict[str, Type["IPipeline"]] = {}


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


class Block(IBlock):
    data: Optional[IData]
    training_workspace: Optional[str]
    serialize_folder: Optional[str]
    previous: Dict[str, "Block"]

    # optional callbacks

    def process_defaults(self, _defaults: OrderedDict) -> None:
        pass

    def run(self, data: IData, _defaults: OrderedDict, **kwargs: Any) -> None:
        pass

    def save_extra(self, folder: str) -> None:
        pass

    def load_from(self, folder: str) -> None:
        pass

    # api

    @property
    def ddp(self) -> bool:
        return get_ddp_info() is not None

    @property
    def local_rank(self) -> Optional[int]:
        ddp_info = get_ddp_info()
        return None if ddp_info is None else ddp_info.local_rank

    @property
    def is_local_rank_0(self) -> bool:
        return not self.ddp or self.local_rank == 0


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

    def to_info(self) -> Dict[str, Any]:
        using_serializable_blocks = self.using_serializable_blocks
        return dict(
            blocks=[
                b.to_pack().asdict() if using_serializable_blocks else b.__identifier__
                for b in self.blocks
            ],
            config=self.config.to_pack().asdict(),
        )

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


class Pipeline(IPipeline):
    data: Optional[IData] = None
    training_workspace: Optional[str] = None
    serialize_folder: Optional[str] = None
    config: DLConfig
    blocks: List[Block]
    _defaults: OrderedDict
    config_file = "config.json"

    # inheritance

    @classmethod
    def init(cls: Type[TPipeline], config: DLConfig) -> TPipeline:
        config.sanity_check()
        self: Pipeline = cls()
        self.config = config.copy()
        self._defaults = OrderedDict()
        return self

    @property
    def config_base(self) -> Type[DLConfig]:
        return DLConfig

    @property
    def block_base(self) -> Type[Block]:
        return Block

    def to_info(self) -> Dict[str, Any]:
        info = super().to_info()
        info["_defaults"] = [[k, v] for k, v in self._defaults.items()]
        return info

    def from_info(self, info: Dict[str, Any]) -> None:
        self._defaults = OrderedDict()
        for k, v in info["_defaults"]:
            self._defaults[k] = v
        super().from_info(info)

    def before_block_build(self, block: Block) -> None:
        block.data = self.data
        block.training_workspace = self.training_workspace
        if self.serialize_folder is None:
            block.serialize_folder = None
        else:
            block.serialize_folder = self.serialize_folder

    def after_block_build(self, block: Block) -> None:
        block.process_defaults(self._defaults)
        if self.training_workspace is not None:
            if self.training_workspace != self.config.workspace:
                self.training_workspace = self.config.workspace

    # api

    def run(self, data: IData, **kwargs: Any) -> None:
        if not self.blocks:
            print("no blocks are built, nothing will happen")
            return
        kw = shallow_copy_dict(kwargs)
        kw["data"] = data
        kw["_defaults"] = self._defaults
        all_blocks = self.block_mappings
        for block in self.blocks:
            block.previous = all_blocks
            safe_execute(block.run, kw)
