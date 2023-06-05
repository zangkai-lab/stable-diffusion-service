import math

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Type, Optional

from tools.bases.register import WithRegister
from models.model.trainer import TrainerState


monitor_dict: Dict[str, Type["TrainerMonitor"]] = {}


class TrainerMonitor(WithRegister["TrainerMonitor"], metaclass=ABCMeta):
    d = monitor_dict

    def __init__(self, *args: Any, **kwargs: Any):
        pass

    @abstractmethod
    def snapshot(self, new_score: float) -> bool:
        pass

    @abstractmethod
    def check_terminate(self, new_score: float) -> bool:
        pass

    @abstractmethod
    def punish_extension(self) -> None:
        pass

    def handle_extension(self, state: TrainerState) -> None:
        if state.should_extend_epoch:
            self.punish_extension()
            new_epoch = state.num_epoch + state.extension
            state.num_epoch = min(new_epoch, state.max_epoch)


