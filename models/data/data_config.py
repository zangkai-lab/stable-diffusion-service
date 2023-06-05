from dataclasses import dataclass
from typing import Any, Dict, Optional, Type, List

from tools.bases.serializable import ISerializableDataClass


data_configs: Dict[str, Type["DataConfig"]] = {}
data_processor_configs: Dict[str, Type["DataProcessorConfig"]] = {}


@dataclass
class DataConfig(ISerializableDataClass):
    for_inference: bool = False
    batch_size: int = 1
    valid_batch_size: Optional[int] = None
    shuffle_train: bool = True
    shuffle_valid: bool = False

    @classmethod
    def d(cls) -> Dict[str, Type["DataConfig"]]:
        return data_configs


@dataclass
class DataProcessorConfig(ISerializableDataClass):
    block_names: Optional[List[str]] = None
    block_configs: Optional[Dict[str, Dict[str, Any]]] = None

    @classmethod
    def d(cls) -> Dict[str, Type["DataProcessorConfig"]]:
        return data_processor_configs

    @property
    def default_blocks(self) -> List[IDataBlock]:
        return []

    def add_blocks(self, *blocks: IDataBlock) -> None:
        if self.block_names is None:
            self.block_names = []
        for b in blocks:
            b_id = b.__identifier__
            if b_id in self.block_names:
                print(f"block `{b_id}` already exists, it will be skipped")
            self.block_names.append(b_id)
            if isinstance(b, INoInitDataBlock):
                continue
            if self.block_configs is None:
                self.block_configs = {}
            self.block_configs[b_id] = b.to_info()

    def set_blocks(self, *blocks: IDataBlock) -> None:
        self.block_names = []
        self.add_blocks(*blocks)
