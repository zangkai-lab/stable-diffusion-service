from typing import Any, Dict, List, Optional, OrderedDict, Type

from .utils import shallow_copy_dict
from .mixins import IBlock
from .mixins import IData
from .utils import safe_execute


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