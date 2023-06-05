import numpy as np

from typing import Dict, Any, List, Optional, Type, TypeVar

from models.data.data_config import DataProcessorConfig

from tools.bases.pipeline import IPipeline
from tools.utils.type import np_dict_type
from tools.utils.safe import safe_execute

from models.data.data_block import IDataBlock
from models.data.data_bundle import DataBundle


TDataProcessor = TypeVar("TDataProcessor", bound="DataProcessor", covariant=True)


@IPipeline.register("base.data_processor")
class DataProcessor(IPipeline):
    config: DataProcessorConfig
    blocks: List[IDataBlock]
    is_ready: bool = False

    # inheritance

    @classmethod
    def init(
        cls: Type[TDataProcessor],
        config: Optional[DataProcessorConfig],
    ) -> TDataProcessor:
        self: DataProcessor = cls()
        self.config = (config or self.config_base()).copy()
        if self.config.block_names is None:
            self.config.set_blocks(*self.config.default_blocks)
        self.before_build_in_init()
        self.build(*(IDataBlock.get(name)() for name in self.config.block_names))  # type: ignore
        return self

    # optional callbacks

    @property
    def config_base(self) -> Type[DataProcessorConfig]:
        return DataProcessorConfig

    @property
    def block_base(self) -> Type[IDataBlock]:
        return IDataBlock

    def before_build_in_init(self) -> None:
        pass

    def after_load(self) -> None:
        self.is_ready = True

    # api

    def _run(self, fn: str, bundle: DataBundle, for_inference: bool) -> DataBundle:
        kw = dict(bundle=bundle.copy(), for_inference=for_inference)
        previous: Dict[str, IDataBlock] = {}
        for block in self.blocks:
            block.previous = previous
            kw["bundle"] = safe_execute(getattr(block, fn), kw)
            previous[block.__identifier__] = block
        return kw["bundle"]  # type: ignore

    def transform(self, bundle: DataBundle, *, for_inference: bool) -> DataBundle:
        return self._run("transform", bundle, for_inference)

    def fit_transform(self, bundle: DataBundle) -> DataBundle:
        bundle = self._run("fit_transform", bundle, False)
        self.is_ready = True
        return bundle

    # changes can happen inplace
    def postprocess_item(self, item: Any) -> np_dict_type:
        for block in self.blocks:
            item = block.postprocess_item(item)
        return item

    def recover_labels(self, y: np.ndarray) -> np.ndarray:
        for block in self.blocks[::-1]:
            y = block.recover_labels(y)
        return y