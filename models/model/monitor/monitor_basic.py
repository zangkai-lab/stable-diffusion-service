import math

from typing import Optional
from models.model.monitor.monitor_train import TrainerMonitor


@TrainerMonitor.register("basic")
class BasicMonitor(TrainerMonitor):
    def __init__(self, patience: int = 25):  # type: ignore
        super().__init__()
        self.patience = patience
        self.num_snapshot = 0
        self.best_score = -math.inf
        self.worst_score: Optional[float] = None

    def snapshot(self, new_score: float) -> bool:
        self.num_snapshot += 1
        if self.worst_score is None:
            self.worst_score = new_score
        else:
            self.worst_score = min(new_score, self.worst_score)
        if new_score > self.best_score:
            self.best_score = new_score
            return True
        return False

    def check_terminate(self, new_score: float) -> bool:
        if self.num_snapshot <= self.patience:
            return False
        if self.worst_score is None:
            return False
        return new_score <= self.worst_score

    def punish_extension(self) -> None:
        return None