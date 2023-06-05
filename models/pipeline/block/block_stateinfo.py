import os
import math

from dataclasses import dataclass

from models.config.train_config import DLConfig
from models.pipeline.pipeline_base import Block
from models.pipeline.block.block_tryload import TryLoadBlock

from tools.bases.dataclass import DataClassBase
from tools.bases.serializable import Serializer


@dataclass
class StateInfo(DataClassBase):
    batch_size: int
    num_batches: int
    num_samples: int
    snapshot_start_step: int
    num_step_per_snapshot: int


@Block.register("extract_state_info")
class ExtractStateInfoBlock(TryLoadBlock):
    config: DLConfig
    state_info: StateInfo

    def try_load(self, folder: str) -> bool:
        info = Serializer.try_load_info(folder)
        if info is None:
            return False
        self.state_info = StateInfo(**info)
        return True

    def from_scratch(self, config: DLConfig) -> None:
        if self.data is None:
            raise ValueError(f"`data` should be provided for `ExtractStateInfoBlock`")
        # from loader
        loader = self.data.get_loaders()[0]
        batch_size = loader.batch_size
        num_batches = len(loader)
        num_samples = len(loader.dataset)
        # from config
        log_steps = config.log_steps
        state_config = config.state_config or {}
        # check log_steps
        if log_steps is not None:
            state_config.setdefault("num_step_per_log", log_steps)
            state_config.setdefault("snapshot_start_step", log_steps)
            state_config.setdefault("num_step_per_snapshot", log_steps)
        # check snapshot_start_step
        snapshot_start_step = state_config.get("snapshot_start_step")
        if snapshot_start_step is None:
            min_num_sample = state_config.get("min_num_sample", 3000)
            snapshot_start_step = math.ceil(min_num_sample / batch_size)
        # check num_step_per_snapshot
        num_step_per_snapshot = state_config.get("num_step_per_snapshot")
        if num_step_per_snapshot is None:
            num_snapshot_per_epoch = state_config.get("num_snapshot_per_epoch", 2)
            max_step_per_snapshot = state_config.get("max_step_per_snapshot", 1000)
            num_step_per_snapshot = max(1, int(num_batches / num_snapshot_per_epoch))
            num_step_per_snapshot = min(
                max_step_per_snapshot,
                num_step_per_snapshot,
            )
        # construct
        state_config["num_step_per_snapshot"] = num_step_per_snapshot
        state_config["snapshot_start_step"] = snapshot_start_step
        self.state_info = StateInfo(
            batch_size=batch_size,
            num_batches=num_batches,
            num_samples=num_samples,
            snapshot_start_step=snapshot_start_step,
            num_step_per_snapshot=num_step_per_snapshot,
        )
        config.state_config = state_config

    def dump_to(self, folder: str) -> None:
        if self.is_local_rank_0:
            Serializer.save_info(folder, info=self.state_info.asdict())