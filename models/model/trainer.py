import torch.nn as nn
import torch

from abc import ABC, abstractmethod, ABCMeta
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from accelerate import Accelerator
from typing import Any, Dict, List, Optional, Set, Tuple, Callable, Type, NamedTuple

from models.model.model_dl import IDLModel
from models.model.train_state import TrainerState
from models.model.loss import ILoss
from models.model.metrics import IMetric, MetricsOutputs
from models.model.tqdm_setting import TqdmSettings
from models.model.inference import IInference
from models.data.data_loader import IDataLoader
from models.data.data import IData

from tools.utils.type import tensor_dict_type
from tools.bases.register import WithRegister
from tools.utils.ddp import is_local_rank_0


monitor_dict: Dict[str, Type["TrainerMonitor"]] = {}
callback_dict: Dict[str, Type["TrainerCallback"]] = {}


class StepOutputs(NamedTuple):
    forward_results: tensor_dict_type
    loss_dict: Dict[str, float]


class MonitorResults(NamedTuple):
    terminate: bool
    save_checkpoint: bool
    metric_outputs: Optional[MetricsOutputs]


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


class TrainerCallback(WithRegister["TrainerCallback"]):
    d = callback_dict

    def __init__(self, *args: Any, **kwargs: Any):
        pass

    @property
    def is_local_rank_0(self) -> bool:
        return is_local_rank_0()

    def initialize(self) -> None:
        pass

    def mutate_train_forward_kwargs(
        self,
        kwargs: Dict[str, Any],
        trainer: "ITrainer",
    ) -> None:
        pass

    def mutate_train_loss_kwargs(
        self,
        kwargs: Dict[str, Any],
        trainer: "ITrainer",
    ) -> None:
        pass

    def before_loop(self, trainer: "ITrainer") -> None:
        pass

    def log_lr(self, key: str, lr: float, state: TrainerState) -> None:
        pass

    def log_metrics(self, metric_outputs: MetricsOutputs, state: TrainerState) -> None:
        pass

    def log_metrics_msg(
        self,
        metric_outputs: MetricsOutputs,
        metrics_log_path: str,
        state: TrainerState,
    ) -> None:
        pass

    def log_artifacts(self, trainer: "ITrainer") -> None:
        pass

    def after_step(self, step_outputs: StepOutputs, state: TrainerState) -> None:
        pass

    def after_monitor(
        self,
        monitor_results: MonitorResults,
        state: TrainerState,
    ) -> None:
        pass

    def finalize(self, trainer: "ITrainer") -> None:
        pass


class ITrainer(ABC):
    config: "TrainerConfig"

    loss: ILoss
    model: IDLModel
    metrics: Optional[IMetric]
    monitors: List[TrainerMonitor]
    callbacks: List[TrainerCallback]
    optimizers: Dict[str, Optimizer]
    schedulers: Dict[str, Optional[_LRScheduler]]
    model_for_training: nn.Module
    accelerator: Accelerator

    state: TrainerState
    train_loader: IDataLoader
    train_loader_copy: IDataLoader
    valid_loader: Optional[IDataLoader]
    inference: IInference

    tqdm_settings: TqdmSettings

    @property
    @abstractmethod
    def export_config(self) -> Dict[str, Any]:
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        pass

    @property
    @abstractmethod
    def use_tqdm_in_validation(self) -> bool:
        pass

    @property
    @abstractmethod
    def validation_loader(self) -> IDataLoader:
        pass

    @property
    @abstractmethod
    def input_sample(self) -> tensor_dict_type:
        pass

    @property
    @abstractmethod
    def workspace(self) -> str:
        pass

    @property
    @abstractmethod
    def checkpoint_folder(self) -> str:
        pass

    @property
    @abstractmethod
    def has_checkpoint_folder(self) -> bool:
        pass

    # init

    @abstractmethod
    def _init_finetune(self) -> None:
        pass

    # core

    @abstractmethod
    def post_loss_step(self, loss_dict: tensor_dict_type) -> None:
        pass

    @abstractmethod
    def clip_norm_step(self) -> None:
        pass

    @abstractmethod
    def optimizer_step(self) -> None:
        pass

    @abstractmethod
    def scheduler_step(self) -> None:
        pass

    @abstractmethod
    def _get_scheduler_settings(
        self,
        key: str,
        scheduler: Any,
    ) -> Tuple[bool, Dict[str, Any]]:
        pass

    @abstractmethod
    def _logging_step(self, metrics_outputs: MetricsOutputs) -> None:
        pass

    @abstractmethod
    def _monitor_step(self, step_outputs: StepOutputs) -> MonitorResults:
        pass

    @abstractmethod
    def _step(self, batch_idx: int, batch: tensor_dict_type) -> StepOutputs:
        pass

    @abstractmethod
    def fit(
        self,
        data: IData,
        loss: ILoss,
        model: IDLModel,
        metrics: Optional[IMetric],
        inference: IInference,
        optimizers: Dict[str, Optimizer],
        schedulers: Dict[str, Optional[_LRScheduler]],
        monitors: List[TrainerMonitor],
        callbacks: List[TrainerCallback],
        schedulers_requires_metric: Set[str],
        *,
        config_export_file: Optional[str] = None,
        show_summary: Optional[bool] = None,
        cuda: Optional[str] = None,
    ) -> "ITrainer":
        pass

    @abstractmethod
    def save_checkpoint(
        self,
        score: float,
        folder: Optional[str] = None,
        *,
        no_history: bool = False,
    ) -> None:
        pass

    @abstractmethod
    def restore_checkpoint(
        self,
        folder: Optional[str] = None,
        strict: bool = True,
        state_dict_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> bool:
        pass