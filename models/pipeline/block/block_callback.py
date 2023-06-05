from typing import List
from models.pipeline.pipeline_base import Block
from models.config.train_config import DLConfig
from models.model.trainer import TrainerCallback

from models.model.callback.callback_log import LogMetricsMsgCallback


@Block.register("build_callbacks")
class BuildCallbacksBlock(Block):
    callbacks: List[TrainerCallback]

    def build(self, config: DLConfig) -> None:
        cb_names = config.callback_names
        cb_configs = config.callback_configs
        use_tqdm = (config.tqdm_settings or {}).get("use_tqdm", False)
        if cb_names is None:
            self.callbacks = [LogMetricsMsgCallback(not use_tqdm)]
        else:
            self.callbacks = TrainerCallback.make_multiple(cb_names, cb_configs)
        for callback in self.callbacks:
            callback.initialize()