import math

from typing import Any, Dict, Optional

from models.data.data_loader import IDataLoader
from tools.utils.ddp import get_world_size
from tools.handler.context import context_error_handler


class TrainerState:
    def __init__(
        self,
        loader: IDataLoader,
        *,
        num_epoch: int,
        max_epoch: int,
        fixed_steps: Optional[int] = None,
        extension: int = 5,
        enable_logging: bool = True,
        min_num_sample: int = 3000,
        snapshot_start_step: Optional[int] = None,
        max_snapshot_file: int = 5,
        num_snapshot_per_epoch: int = 2,
        num_step_per_log: int = 350,
        num_step_per_snapshot: Optional[int] = None,
        max_step_per_snapshot: int = 1000,
        min_snapshot_epoch_gap: int = 0,
    ):
        self.step = self.epoch = 0
        self.batch_size = loader.batch_size * get_world_size()
        self.num_step_per_epoch = len(loader)
        self.num_epoch = num_epoch
        self.max_epoch = max_epoch
        self.fixed_steps = fixed_steps
        self.extension = extension
        self.enable_logging = enable_logging
        self.min_num_sample = min_num_sample
        if snapshot_start_step is None:
            snapshot_start_step = math.ceil(min_num_sample / self.batch_size)
        self.snapshot_start_step = snapshot_start_step
        self.max_snapshot_file = max_snapshot_file
        self.num_snapshot_per_epoch = num_snapshot_per_epoch
        self.num_step_per_log = num_step_per_log
        if num_step_per_snapshot is None:
            num_step_per_snapshot = max(1, int(len(loader) / num_snapshot_per_epoch))
            num_step_per_snapshot = min(max_step_per_snapshot, num_step_per_snapshot)
        self.num_step_per_snapshot = num_step_per_snapshot
        self.max_step_per_snapshot = max_step_per_snapshot
        self.min_snapshot_epoch_gap = min_snapshot_epoch_gap
        self._previous_snapshot_epoch = 0

    def set_terminate(self) -> None:
        self.step = self.epoch = -1

    def update_snapshot_epoch(self) -> None:
        self._previous_snapshot_epoch = self.epoch

    @property
    def config(self) -> Dict[str, Any]:
        return {
            "num_epoch": self.num_epoch,
            "max_epoch": self.max_epoch,
            "fixed_steps": self.fixed_steps,
            "extension": self.extension,
            "enable_logging": self.enable_logging,
            "min_num_sample": self.min_num_sample,
            "snapshot_start_step": self.snapshot_start_step,
            "max_snapshot_file": self.max_snapshot_file,
            "num_snapshot_per_epoch": self.num_snapshot_per_epoch,
            "num_step_per_log": self.num_step_per_log,
            "num_step_per_snapshot": self.num_step_per_snapshot,
            "max_step_per_snapshot": self.max_step_per_snapshot,
        }

    @property
    def is_terminate(self) -> bool:
        return self.epoch == -1

    @property
    def should_train(self) -> bool:
        if self.fixed_steps is not None:
            return self.step < self.fixed_steps
        return self.epoch < self.num_epoch

    @property
    def should_terminate(self) -> bool:
        if self.fixed_steps is None:
            return False
        return self.step == self.fixed_steps

    @property
    def should_monitor(self) -> bool:
        return self.step % self.num_step_per_snapshot == 0

    @property
    def should_log_lr(self) -> bool:
        if not self.enable_logging:
            return False
        denominator = min(self.num_step_per_epoch, 10)
        return self.step % denominator == 0

    @property
    def should_log_losses(self) -> bool:
        if not self.enable_logging:
            return False
        patience = max(4, int(round(self.num_step_per_epoch / 50.0)))
        denominator = min(self.num_step_per_epoch, patience)
        return self.step % denominator == 0

    @property
    def should_log_artifacts(self) -> bool:
        return self.should_log_metrics_msg

    @property
    def should_log_metrics_msg(self) -> bool:
        if not self.enable_logging:
            return False
        if self.is_terminate:
            return True
        min_period = math.ceil(self.num_step_per_log / self.num_step_per_snapshot)
        period = max(1, int(min_period)) * self.num_step_per_snapshot
        return self.step % period == 0

    @property
    def can_snapshot(self) -> bool:
        if self.is_terminate:
            return True
        return self.epoch - self._previous_snapshot_epoch >= self.min_snapshot_epoch_gap

    @property
    def should_start_snapshot(self) -> bool:
        return self.step >= self.snapshot_start_step

    @property
    def should_extend_epoch(self) -> bool:
        return self.epoch == self.num_epoch and self.epoch < self.max_epoch

    @property
    def reached_max_epoch(self) -> bool:
        return self.epoch > self.max_epoch

    @property
    def disable_logging(self) -> context_error_handler:
        class _(context_error_handler):
            def __init__(self, state: TrainerState):
                self.state = state
                self.enabled = state.enable_logging

            def __enter__(self) -> None:
                self.state.enable_logging = False

            def _normal_exit(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                self.state.enable_logging = self.enabled

        return _(self)