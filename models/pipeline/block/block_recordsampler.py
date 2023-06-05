from typing import Any
from collections import OrderedDict

from models.pipeline.pipeline_base import Block
from models.config.train_config import DLConfig
from models.data.data import IData


@Block.register("record_num_samples")
class RecordNumSamplesBlock(Block):
    def build(self, config: DLConfig) -> None:
        pass

    def run(self, data: IData, _defaults: OrderedDict, **kwargs: Any) -> None:
        _defaults["train_samples"] = len(data.train_dataset)
        if data.valid_dataset is None:
            _defaults["valid_samples"] = None
        else:
            _defaults["valid_samples"] = len(data.valid_dataset)
        _defaults.move_to_end("valid_samples", last=False)
        _defaults.move_to_end("train_samples", last=False)