from dataclasses import dataclass
from typing import Any, Dict, Optional, Type, Union, List

from tools.utils.type import configs_type
from tools.utils.safe import safe_execute
from tools.enum.precision import PrecisionType
from tools.bases.serializable import ISerializableDataClass
from tools.bases.dataclass import DataClassBase


trainer_configs: Dict[str, Type["TrainerConfig"]] = {}


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
    def sanity_check(self) -> None:  # 安全检测
        if not self.model_name:
            raise ValueError("`model_name` should be provided")



@dataclass
class MLEncoderSettings(DataClassBase):
    """
    Encoder settings.

    Properties
    ----------
    dim (int) : number of different values of this categorical column.
    methods (str | List[str]) : encoding methods to use for each categorical column.
        * if List[str] is provided and its length > 1, then multiple encoding methods will be used.
    method_configs (Dict[str, Any]) : (flattened) configs of the corresponding encoding methods.
        * even if multiple methods are used, `method_configs` should still be 'flattened'

    """

    dim: int
    methods: Union[str, List[str]] = "embedding"
    method_configs: Optional[Dict[str, Any]] = None

    @property
    def use_one_hot(self) -> bool:
        if self.methods == "one_hot":
            return True
        if isinstance(self.methods, list) and "one_hot" in self.methods:
            return True
        return False

    @property
    def use_embedding(self) -> bool:
        if self.methods == "embedding":
            return True
        if isinstance(self.methods, list) and "embedding" in self.methods:
            return True
        return False


@dataclass
class MLGlobalEncoderSettings(DataClassBase):
    embedding_dim: Optional[int] = None
    embedding_dropout: Optional[float] = None


@dataclass
@Config.register("ml")
class MLConfig(DLConfig):
    """
    * encoder_settings: used by `Encoder`.
    * global_encoder_settings: used by `Encoder`.
    * index_mapping: since there might be some redundant columns, we may need to
    map the original keys of the `encoder_settings` to the new ones.
    * infer_encoder_settings: whether infer the `encoder_settings` based on
    information gathered by `RecognizerBlock`.
    """

    encoder_settings: Optional[Dict[str, MLEncoderSettings]] = None
    global_encoder_settings: Optional[MLGlobalEncoderSettings] = None
    index_mapping: Optional[Dict[str, int]] = None
    infer_encoder_settings: bool = True

    def from_info(self, info: Dict[str, Any]) -> None:
        super().from_info(info)
        if self.encoder_settings is not None:
            self.encoder_settings = {
                str_idx: MLEncoderSettings(**settings)
                for str_idx, settings in self.encoder_settings.items()
            }
        ges = self.global_encoder_settings
        if ges is not None:
            self.global_encoder_settings = MLGlobalEncoderSettings(**ges)