from dataclasses import dataclass
from typing import Any, Dict, Optional, Union, List, Type
from enum import Enum

from .serializable import ISerializableDataClass

from ..utils import safe_execute


configs_type = Optional[Union[List[Dict[str, Any]], Dict[str, Any]]]
trainer_configs: Dict[str, Type["TrainerConfig"]] = {}


class PrecisionType(str, Enum):
    NO = "no"
    FP8 = "fp8"
    FP16 = "fp16"
    BF16 = "bf16"


@dataclass
class TrainerConfig(ISerializableDataClass):
    state_config: Optional[Dict[str, Any]] = None
    workspace: str = "_logs"
    create_sub_workspace: bool = True
    num_epoch: int = 40
    max_epoch: int = 1000
    fixed_epoch: Optional[int] = None
    fixed_steps: Optional[int] = None
    log_steps: Optional[int] = None
    valid_portion: float = 1.0
    mixed_precision: Union[str, PrecisionType] = PrecisionType.NO
    clip_norm: float = 0.0
    metric_names: Optional[Union[str, List[str]]] = None
    metric_configs: configs_type = None
    metric_weights: Optional[Dict[str, float]] = None
    use_losses_as_metrics: Optional[bool] = None
    loss_metrics_weights: Optional[Dict[str, float]] = None
    recompute_train_losses_in_eval: bool = True
    monitor_names: Optional[Union[str, List[str]]] = None
    monitor_configs: Optional[Dict[str, Any]] = None
    auto_callback: bool = True
    callback_names: Optional[Union[str, List[str]]] = None
    callback_configs: Optional[Dict[str, Any]] = None
    lr: Optional[float] = None
    optimizer_name: Optional[str] = None
    scheduler_name: Optional[str] = None
    optimizer_config: Optional[Dict[str, Any]] = None
    scheduler_config: Optional[Dict[str, Any]] = None
    update_scheduler_per_epoch: bool = False
    optimizer_settings: Optional[Dict[str, Dict[str, Any]]] = None
    use_zero: bool = False
    finetune_config: Optional[Dict[str, Any]] = None
    tqdm_settings: Optional[Dict[str, Any]] = None

    @classmethod
    def d(cls) -> Dict[str, Type["TrainerConfig"]]:
        return trainer_configs


@dataclass
class Config(TrainerConfig):
    loss_name: Optional[str] = None
    loss_config: Optional[Dict[str, Any]] = None
    in_loading: bool = False
    allow_no_loss: bool = False
    cudnn_benchmark: bool = False

    def to_debug(self) -> None:
        self.fixed_steps = 1
        self.valid_portion = 1.0e-4

    @property
    def trainer_config(self) -> TrainerConfig:
        return safe_execute(TrainerConfig, self.asdict())


@dataclass
class _DLConfig:
    model_name: str = ""
    model_config: Optional[Dict[str, Any]] = None
    num_repeat: Optional[int] = None
    inference_type: str = "dl"


@dataclass
@Config.register("dl")
class DLConfig(Config, _DLConfig):
    def sanity_check(self) -> None:
        if not self.model_name:
            raise ValueError("`model_name` should be provided")