import os
import json
import re
import tqdm
import math
import torch

from accelerate import Accelerator
from torch import Tensor
from torch import nn
from torch.optim import Optimizer
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import _LRScheduler
from abc import ABCMeta, abstractmethod, ABC
from typing import Any, Dict, List, Optional, Tuple, Union, NamedTuple, Callable, Set, Type
from dataclasses import dataclass

from .config import TrainerConfig
from .withRegister import WithRegister
from .idata import IData
from .idata import DataClassBase

from ..block import ILoss
from ..block import IMetric
from ..block import IInference
from ..block import IDLModel
from ..block import IDataLoader
from ..block import TrainerState
from ..block import MetricsOutputs
from ..block import PrecisionType
from ..block import eval_context
from ..block import get_device
from ..block import MultipleMetrics

from ..utils import is_local_rank_0
from ..utils import to_device
from ..utils import np_batch_to_tensor
from ..utils import shallow_copy_dict
from ..utils import safe_execute
from ..utils import get_ddp_info


LOSS_KEY = "loss"
INPUT_KEY = "input"
LATENT_KEY = "latent"
PREDICTIONS_KEY = "predictions"
LABEL_KEY = "labels"

PT_PREFIX = "model_"
SCORES_FILE = "scores.json"
CHECKPOINTS_FOLDER = "checkpoints"

tensor_dict_type = Dict[str, Union[torch.Tensor, Any]]
monitor_dict: Dict[str, Type["TrainerMonitor"]] = {}
callback_dict: Dict[str, Type["TrainerCallback"]] = {}


def weighted_loss_score(config: TrainerConfig, loss_items: Dict[str, float]) -> float:
    if not config.loss_metrics_weights:
        loss = loss_items.get(LOSS_KEY)
        if loss is not None:
            return -loss
        return -sum(loss_items.values()) / len(loss_items)
    score = 0.0
    for k, w in config.loss_metrics_weights.items():
        v = loss_items.get(k)
        if v is None:
            continue
        score -= v * w
    return score


class StepOutputs(NamedTuple):
    forward_results: tensor_dict_type
    loss_dict: Dict[str, float]


class CustomTrainStepLoss(NamedTuple):
    loss: Tensor
    losses: Dict[str, float]


class CustomTrainStep(ABC):
    def __init__(
        self,
        scope: str = "all",
        *,
        num_forward: int = 1,
        grad_accumulate: int = 1,
        requires_new_forward: bool = False,
        requires_grad_in_forward: bool = True,
        requires_scheduler_step: bool = False,
        enable_toggle_optimizer: bool = True,
    ) -> None:
        self.scope = scope
        self.num_forward = num_forward
        self.grad_accumulate = grad_accumulate
        self.requires_new_forward = requires_new_forward
        self.requires_grad_in_forward = requires_grad_in_forward
        self.requires_scheduler_step = requires_scheduler_step
        self.enable_toggle_optimizer = enable_toggle_optimizer

    # abstract

    @abstractmethod
    def loss_fn(
        self,
        m: "ModelWithCustomSteps",
        state: Optional[TrainerState],
        batch: tensor_dict_type,
        forward_results: Union[tensor_dict_type, List[tensor_dict_type]],
        **kwargs: Any,
    ) -> CustomTrainStepLoss:
        pass

    # optional callbacks

    def should_skip(self, m: "ModelWithCustomSteps", state: TrainerState) -> bool:
        return False

    def callback(
        self,
        m: "ModelWithCustomSteps",
        trainer: "ITrainer",
        batch: tensor_dict_type,
        forward_results: Union[tensor_dict_type, List[tensor_dict_type]],
    ) -> None:
        pass


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


class MonitorResults(NamedTuple):
    terminate: bool
    save_checkpoint: bool
    metric_outputs: Optional[MetricsOutputs]


def get_input_sample(loader: IDataLoader, device: torch.device) -> tensor_dict_type:
    sample = next(iter(TensorBatcher(loader, device)))
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            sample[k] = v[:1]
        elif isinstance(v, list):
            sample[k] = [vv[:1] if isinstance(vv, torch.Tensor) else vv for vv in v]
        else:
            sample[k] = v
    return sample


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


@dataclass
class TqdmSettings(DataClassBase):
    use_tqdm: bool = False
    use_step_tqdm: bool = False
    use_tqdm_in_validation: bool = False
    in_distributed: bool = False
    position: int = 0
    desc: str = "epoch"


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


def get_update_fn(trainer: ITrainer) -> Callable[[Tensor, Optimizer, bool], None]:
    def update_fn(loss: Tensor, optimizer: Optimizer, update: bool) -> None:
        accelerator = trainer.accelerator
        accelerator.backward(loss)
        if update:
            trainer.clip_norm_step()
            optimizer.step()
            optimizer.zero_grad()

    return update_fn


class no_grad_context(torch.no_grad):
    def __init__(self, *, enabled: bool):
        super().__init__()
        self.enabled = enabled

    def __enter__(self) -> None:
        if not self.enabled:
            return
        super().__enter__()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if not self.enabled:
            return
        super().__exit__(exc_type, exc_val, exc_tb)


class toggle_optimizer:
    def __init__(self, m: nn.Module, optimizer: Optimizer, *, enabled: bool = True):
        self.m = m
        self.optimizer = optimizer
        self.enabled = enabled
        self.requires_grad: Dict[str, bool] = {}

    def __enter__(self) -> None:
        if not self.enabled:
            return
        self.requires_grad = {k: p.requires_grad for k, p in self.m.named_parameters()}
        for p in self.m.parameters():
            p.requires_grad = False
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                p.requires_grad = True

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if not self.enabled:
            return
        for k, p in self.m.named_parameters():
            requires_grad = self.requires_grad.get(k)
            if requires_grad is not None:
                p.requires_grad = requires_grad


def run_train_steps(
    m: "ModelWithCustomSteps",
    train_steps: List["CustomTrainStep"],
    *,
    batch_idx: int,
    batch: tensor_dict_type,
    trainer: "ITrainer",
    forward_kwargs: Dict[str, Any],
    loss_kwargs: Dict[str, Any],
) -> StepOutputs:
    state = trainer.state
    forward: Union[tensor_dict_type, List[tensor_dict_type]] = {}
    loss_dict = {}
    update_fn = get_update_fn(trainer)
    any_update = False
    performed_scheduler_step = False
    # sanity check
    fw_has_grad = True
    fw_train_step: Any = ()
    for i, train_step in enumerate(train_steps):
        if i == 0 or train_step.requires_new_forward:
            fw_has_grad = train_step.requires_grad_in_forward
            fw_train_step = train_step
        if not fw_has_grad and train_step.requires_grad_in_forward:
            fw_name = fw_train_step.__class__.__name__
            current_name = train_step.__class__.__name__
            raise ValueError(
                f"current forward pass comes from '{fw_name}' and has no grad, "
                f"but '{current_name}' requires grad in forward. You can either set "
                f"`requires_grad_in_forward` of '{fw_name}' to True, or set "
                f"`requires_new_forward` of '{current_name}' to True."
            )
    # run train steps
    get_fw = lambda: m.run(batch_idx, batch, state, **forward_kwargs)
    for i, train_step in enumerate(train_steps):
        if train_step.should_skip(m, state):
            continue
        if i == 0 or train_step.requires_new_forward:
            with no_grad_context(enabled=not train_step.requires_grad_in_forward):
                if train_step.num_forward == 1:
                    forward = get_fw()
                else:
                    forward = [get_fw() for _ in range(train_step.num_forward)]
        optimizer = trainer.optimizers[train_step.scope]
        with toggle_optimizer(m, optimizer, enabled=train_step.enable_toggle_optimizer):
            with autocast(enabled=trainer.config.mixed_precision != PrecisionType.NO):
                loss_res = train_step.loss_fn(m, state, batch, forward, **loss_kwargs)
            update = state.step % train_step.grad_accumulate == 0
            update_fn(loss_res.loss, optimizer, update)
            loss_dict.update(loss_res.losses)
        if update:
            any_update = True
            performed_scheduler_step = train_step.requires_scheduler_step
            if performed_scheduler_step:
                trainer.scheduler_step()
    if any_update and not performed_scheduler_step:
        trainer.scheduler_step()
    # callbacks
    for train_step in train_steps:
        train_step.callback(m, trainer, batch, forward)
    return StepOutputs(forward, loss_dict)


class TensorBatcher:
    def __init__(self, loader: IDataLoader, device: torch.device) -> None:
        self.loader = loader
        self.device = device

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self) -> "TensorBatcher":
        self.loader.__iter__()
        return self

    def __next__(self) -> tensor_dict_type:
        npd = self.loader.__next__()
        batch = np_batch_to_tensor(npd)
        return to_device(batch, self.device)

    def to(self, device: torch.device) -> None:
        self.device = device

    def get_full_batch(self) -> tensor_dict_type:
        return np_batch_to_tensor(self.loader.get_full_batch())


class ModelWithCustomSteps(IDLModel, metaclass=ABCMeta):
    custom_train_step: bool = True
    custom_evaluate_step: bool = True
    custom_params_groups: bool = False
    custom_ddp_initialization: bool = False

    # abstract

    @property
    @abstractmethod
    def train_steps(self) -> List[CustomTrainStep]:
        pass

    @abstractmethod
    def evaluate(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional["TrainerState"],
        weighted_loss_score_fn: Callable[[Dict[str, float]], float],
        forward_kwargs: Dict[str, Any],
    ) -> MetricsOutputs:
        pass

    # inheritance

    def permute_trainer_config(self, trainer_config: "TrainerConfig") -> None:
        if trainer_config.optimizer_settings is None:
            opt_settings = {}
            for step in self.train_steps:
                scope = step.scope
                opt_settings[scope] = dict(
                    optimizer=trainer_config.optimizer_name or "adam",
                    scheduler=trainer_config.scheduler_name,
                    optimizer_config=trainer_config.optimizer_config,
                    scheduler_config=trainer_config.scheduler_config,
                )
            trainer_config.optimizer_settings = opt_settings

    # optional callback

    def params_groups(self, m: nn.Module) -> List[Dict[str, Any]]:
        return []

    # api

    def train_step(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        trainer: "ITrainer",
        forward_kwargs: Dict[str, Any],
        loss_kwargs: Dict[str, Any],
    ) -> StepOutputs:
        return run_train_steps(
            self,
            self.train_steps,
            batch_idx=batch_idx,
            batch=batch,
            trainer=trainer,
            forward_kwargs=forward_kwargs,
            loss_kwargs=loss_kwargs,
        )

    def evaluate_step(
        self,
        config: "TrainerConfig",
        loader: IDataLoader,
        portion: float,
        state: Optional["TrainerState"],
        forward_kwargs: Optional[Dict[str, Any]] = None,
    ) -> MetricsOutputs:
        loss_score_fn = lambda loss_items: weighted_loss_score(config, loss_items)
        device = get_device(self)
        tensor_batcher = TensorBatcher(loader, device)
        is_positive: Dict[str, bool] = {}
        metric_values: Dict[str, List[float]] = {}
        final_scores = []
        for i, batch in enumerate(tensor_batcher):
            if i / len(tensor_batcher) >= portion:
                break
            out = self.evaluate(i, batch, state, loss_score_fn, forward_kwargs or {})
            final_scores.append(out.final_score)
            for k, v in out.metric_values.items():
                metric_values.setdefault(k, []).append(v)
                is_positive[k] = out.is_positive[k]
        return MetricsOutputs(
            sum(final_scores) / len(final_scores),
            {k: sum(v) / len(v) for k, v in metric_values.items()},
            is_positive,
        )


def get_metrics(
    config: TrainerConfig,
    model: IDLModel,
    loss: ILoss,
    metrics: Optional[IMetric],
    inference: IInference,
    loader: IDataLoader,
    *,
    portion: float = 1.0,
    state: Optional[TrainerState] = None,
    forward_kwargs: Optional[Dict[str, Any]] = None,
) -> MetricsOutputs:
    if isinstance(model, ModelWithCustomSteps) and model.custom_evaluate_step:
        use_grad = inference.use_grad_in_predict
        args = config, loader, portion, state, forward_kwargs
        try:
            with eval_context(model, use_grad=use_grad):
                rs = model.evaluate_step(*args)
        except:
            inference.use_grad_in_predict = True
            with eval_context(model, use_grad=True):
                rs = model.evaluate_step(*args)
        return rs
    outputs = inference.get_outputs(
        loader,
        portion=portion,
        state=state,
        metrics=metrics,
        loss=loss if config.use_losses_as_metrics else None,
        return_outputs=False,
        **(forward_kwargs or {}),
    )
    metric_values = {}
    is_positive = {}
    final_scores = []
    loss_items = outputs.loss_items
    metric_outputs = outputs.metric_outputs
    if loss_items is not None:
        metric_values.update(loss_items)
        is_positive.update({k: False for k in loss_items})
        final_scores.append(weighted_loss_score(config, loss_items))
    if metric_outputs is not None:
        metric_values.update(metric_outputs.metric_values)
        is_positive.update(metric_outputs.is_positive)
        final_scores.append(metric_outputs.final_score)
    final_score = sum(final_scores) / len(final_scores)
    return MetricsOutputs(final_score, metric_values, is_positive)


class Trainer(ITrainer):
    model_log_file = "model.txt"
    metrics_log_file = "metrics.txt"
    summary_log_file = "summary.txt"

    def __init__(self, config: TrainerConfig):
        self.config = config
        self.tqdm_settings = safe_execute(TqdmSettings, config.tqdm_settings or {})
        self._current_scheduler_epoch = -1
        self.lr_metrics_updated = False
        self.intermediate: Optional[MetricsOutputs] = None
        self.final_results: Optional[MetricsOutputs] = None
        self.checkpoint_scores: Dict[str, float] = {}

    @property
    def export_config(self) -> Dict[str, Any]:
        ddp_info = get_ddp_info()
        ddp_d = None if ddp_info is None else ddp_info._asdict()
        return {
            "state_config": self.state.config,
            "valid_portion": self.config.valid_portion,
            "mixed_precision": self.config.mixed_precision,
            "clip_norm": self.config.clip_norm,
            "metrics": (
                None
                if self.metrics is None
                else self.metrics.__identifier__
                if not isinstance(self.metrics, MultipleMetrics)
                else [metric.__identifier__ for metric in self.metrics.metrics]
            ),
            "loss_metrics_weights": self.config.loss_metrics_weights,
            "monitors": [monitor.__identifier__ for monitor in self.monitors],
            "callbacks": [callback.__identifier__ for callback in self.callbacks],
            "optimizer_settings": self.config.optimizer_settings,
            "ddp_info": ddp_d,
            "finetune_config": self.config.finetune_config,
            "tqdm_settings": self.tqdm_settings.asdict(),
        }

    @property
    def device(self) -> torch.device:
        return self.accelerator.device

    @property
    def is_local_rank_0(self) -> bool:
        return self.accelerator.is_local_main_process

    @property
    def use_tqdm_in_validation(self) -> bool:
        if not self.is_local_rank_0:
            return False
        if self.tqdm_settings.in_distributed:
            return False
        return self.tqdm_settings.use_tqdm_in_validation or self.state.is_terminate

    @property
    def validation_loader(self) -> IDataLoader:
        return self.valid_loader or self.train_loader_copy

    @property
    def input_sample(self) -> tensor_dict_type:
        return get_input_sample(self.train_loader_copy, self.device)

    @property
    def has_checkpoint_folder(self) -> bool:
        if self.checkpoint_folder is None:
            return False
        return os.path.isdir(self.checkpoint_folder)

    @property
    def workspace(self) -> str:
        return self.config.workspace

    @property
    def checkpoint_folder(self) -> str:
        return os.path.join(self.workspace, CHECKPOINTS_FOLDER)

    # init

    def _init_finetune(self) -> None:
        finetune_config = self.config.finetune_config
        if finetune_config is None:
            return None
        pretrained_ckpt = finetune_config.get("pretrained_ckpt")
        if pretrained_ckpt is None:
            raise ValueError("`rank` should be provided when `finetune` is triggered")
        print(f"loading pretrained checkpoint from '{pretrained_ckpt}'...")
        d = torch.load(pretrained_ckpt, map_location=self.device)
        self.model.load_state_dict(d)
        freeze = finetune_config.get("freeze", "")
        freeze_except = finetune_config.get("freeze_except", "")
        if not freeze and not freeze_except:
            return None
        msg_fmt = f"-> {'{}'} parameters will be {'{}'} under '{'{}'}'"
        param_names = []
        if freeze:
            num_frozen = 0
            for name, param in self.model.named_parameters():
                if re.match(freeze, name):
                    num_frozen += 1
                    param.requires_grad_(False)
                    param_names.append(name)
            msg = msg_fmt.format(num_frozen, "frozen", freeze)
        elif freeze_except:
            num_trainable = 0
            for name, param in self.model.named_parameters():
                if not re.match(freeze_except, name):
                    param.requires_grad_(False)
                else:
                    num_trainable += 1
                    param_names.append(name)
            msg = msg_fmt.format(num_trainable, "trainable", freeze_except)
        else:
            msg = "`freeze` & `freeze_except` should not be provided simultaneously"
            raise ValueError(msg)
        print("\n".join(["=" * 100, msg, "-" * 100] + param_names + ["-" * 100]))

    # core

    def post_loss_step(self, loss_dict: tensor_dict_type) -> None:
        # backward
        loss = loss_dict[LOSS_KEY]
        self.accelerator.backward(loss)
        # clip norm
        self.clip_norm_step()
        # optimize
        self.optimizer_step()
        self.scheduler_step()

    def weighted_loss_score(self, loss_items: Dict[str, float]) -> float:
        if not self.config.loss_metrics_weights:
            loss = loss_items.get(LOSS_KEY)
            if loss is not None:
                return -loss
            return -sum(loss_items.values()) / len(loss_items)
        score = 0.0
        for k, w in self.config.loss_metrics_weights.items():
            v = loss_items.get(k)
            if v is None:
                continue
            score -= v * w
        return score

    def clip_norm_step(self) -> None:
        if self.config.clip_norm > 0.0:
            if self.accelerator.sync_gradients:
                self._gradient_norm = self.accelerator.clip_grad_norm_(
                    self.model_for_training.parameters(),
                    max_norm=self.config.clip_norm,
                )

    def optimizer_step(self) -> None:
        for opt in self.optimizers.values():
            opt.step()
        for param in self.model_for_training.parameters():
            param.grad = None

    def scheduler_step(self) -> None:
        if self.config.update_scheduler_per_epoch:
            if self.state.epoch == self._current_scheduler_epoch:
                return
        lr_metric_logged = False
        for key, scheduler in self.schedulers.items():
            if scheduler is not None:
                should_log_lr, kwargs = self._get_scheduler_settings(key, scheduler)
                if should_log_lr or self.config.update_scheduler_per_epoch:
                    lr_metric_logged = True
                    for callback in self.callbacks:
                        callback.log_lr(
                            f"lr-{key}",
                            scheduler.get_last_lr()[0],
                            self.state,
                        )
                scheduler.step(**shallow_copy_dict(kwargs))
        if lr_metric_logged:
            self.lr_metrics_updated = False
        if self.config.update_scheduler_per_epoch:
            self._current_scheduler_epoch = self.state.epoch

    def _get_metrics(
        self,
        *,
        portion: float = 1.0,
        loader: Optional[IDataLoader] = None,
    ) -> MetricsOutputs:
        return get_metrics(
            self.config,
            self.model,
            self.loss,
            self.metrics,
            self.inference,
            loader or self.validation_loader,
            portion=portion,
            state=self.state,
        )

    def _get_scheduler_settings(
        self,
        key: str,
        scheduler: Any,
    ) -> Tuple[bool, Dict[str, Any]]:
        kwargs = {}
        should_log_lr = self.state.should_log_lr
        is_warmup = isinstance(scheduler, WarmupScheduler)
        requires_metric = key in self.schedulers_requires_metric
        if requires_metric and not (is_warmup and not scheduler.finished_warmup):
            if self.intermediate is None:
                kwargs["metrics"] = -math.inf
            else:
                kwargs["metrics"] = self.intermediate.final_score
            should_log_lr &= self.lr_metrics_updated
        return should_log_lr, kwargs

    def _logging_step(self, metrics_outputs: MetricsOutputs) -> None:
        if not self.is_local_rank_0:
            return None
        if self.epoch_tqdm is not None:
            metric_values = shallow_copy_dict(metrics_outputs.metric_values)
            metric_values["score"] = metrics_outputs.final_score
            self.epoch_tqdm.set_postfix(metric_values)
        for callback in self.callbacks:
            callback.log_metrics(metrics_outputs, self.state)
        if self.state.should_log_artifacts:
            for callback in self.callbacks:
                callback.log_artifacts(self)
        if self.state.should_log_metrics_msg:
            for callback in self.callbacks:
                callback.log_metrics_msg(
                    metrics_outputs,
                    self.metrics_log_path,
                    self.state,
                )

    def _monitor_step(self, step_outputs: StepOutputs) -> MonitorResults:
        terminate = False
        save_checkpoint = False
        for monitor in self.monitors:
            monitor.handle_extension(self.state)
        if self.state.should_monitor:
            # get metrics
            if (
                self.valid_loader is not None
                or self.config.recompute_train_losses_in_eval
            ):
                self.intermediate = self._get_metrics(portion=self.config.valid_portion)
            else:
                loss_dict = step_outputs.loss_dict
                is_positive = {k: False for k in loss_dict}
                loss_score = weighted_loss_score(self.config, loss_dict)
                self.intermediate = MetricsOutputs(loss_score, loss_dict, is_positive)
            self.lr_metrics_updated = True
            # logging
            self._logging_step(self.intermediate)
            # check terminate
            if self.state.should_start_snapshot:
                score = self.intermediate.final_score
                if any(monitor.snapshot(score) for monitor in self.monitors):
                    if self.state.can_snapshot:
                        self.state.update_snapshot_epoch()
                        save_checkpoint = True
                if any(monitor.check_terminate(score) for monitor in self.monitors):
                    terminate = True
        return MonitorResults(terminate, save_checkpoint, self.intermediate)

    def _step(self, batch_idx: int, batch: tensor_dict_type) -> StepOutputs:
        batch = to_device(batch, self.device)
        # kwargs
        forward_kw: Dict[str, Any] = {}
        for callback in self.callbacks:
            callback.mutate_train_forward_kwargs(forward_kw, self)
        loss_kw: Dict[str, Any] = {}
        for callback in self.callbacks:
            callback.mutate_train_loss_kwargs(loss_kw, self)
        # allow model defines its own training step
        if (
            isinstance(self.model, ModelWithCustomSteps)
            and self.model.custom_train_step
        ):
            return self.model.train_step(batch_idx, batch, self, forward_kw, loss_kw)
        # forward & loss
        forward_results = self.model.run(batch_idx, batch, self.state, **forward_kw)
        loss_dict = self.loss.run(forward_results, batch, self.state, **loss_kw)
        # post loss step
        self.post_loss_step(loss_dict)
        return StepOutputs(forward_results, {k: v.item() for k, v in loss_dict.items()})

    # api

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
    ) -> "Trainer":
        # accelerator
        cpu = False
        if get_ddp_info() is None:
            if cuda is None:
                cpu = True
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = cuda
        if isinstance(self.config.mixed_precision, PrecisionType):
            self.config.mixed_precision = self.config.mixed_precision.value
        self.accelerator = Accelerator(
            cpu=cpu,
            mixed_precision=self.config.mixed_precision,
        )
        # initialize artifact structure
        if self.is_local_rank_0:
            os.makedirs(self.workspace, exist_ok=True)
            self.metrics_log_path = os.path.join(self.workspace, self.metrics_log_file)
            with open(self.metrics_log_path, "w"):
                pass
            os.makedirs(self.checkpoint_folder, exist_ok=True)
        # initialize
        self.model = model
        self.metrics = metrics
        self.monitors = monitors
        self.callbacks = callbacks
        self.schedulers_requires_metric = schedulers_requires_metric
        if self.is_local_rank_0:
            with open(os.path.join(self.workspace, self.model_log_file), "w") as f:
                f.write(str(model))
        self.inference = inference
        # data
        train_loader, valid_loader = data.get_loaders()
        self.train_loader = train_loader
        self.train_loader_copy = train_loader.copy()
        self.train_loader_copy.disable_shuffle()
        self.valid_loader = valid_loader
        if self.config.fixed_epoch is not None:
            num_epoch = max_epoch = self.config.fixed_epoch
        else:
            max_epoch = self.config.max_epoch
            num_epoch = min(max_epoch, self.config.num_epoch)
        self.state = TrainerState(
            train_loader,
            num_epoch=num_epoch,
            max_epoch=max_epoch,
            fixed_steps=self.config.fixed_steps,
            **(self.config.state_config or {}),
        )
        # accelerator prepare
        optimizer_keys = sorted(optimizers)
        (
            self.loss,
            self.model_for_training,
            *optimizer_list,
        ) = self.accelerator.prepare(
            loss,
            model,
            *[optimizers[k] for k in optimizer_keys],
        )
        self.optimizers = {k: optimizer_list[i] for i, k in enumerate(optimizer_keys)}
        self.schedulers = schedulers
        for sch in schedulers.values():
            if sch is not None:
                sch.load_state_dict(sch.state_dict())
        # callback
        self.model.init_with_trainer(self)
        # finetune
        self._init_finetune()
        # verbose
        if show_summary is None:
            show_summary = not self.tqdm_settings.in_distributed
        if self.is_local_rank_0:
            summary_msg = summary(
                self.model,
                to_device(self.input_sample, self.device),
                return_only=not show_summary,
            )
            with open(os.path.join(self.workspace, self.summary_log_file), "w") as f:
                f.write(summary_msg)
        # tqdm
        step_tqdm = None
        self.epoch_tqdm: Optional[tqdm] = None
        if self.is_local_rank_0 and self.tqdm_settings.use_tqdm:
            self.epoch_tqdm = tqdm(
                list(range(self.state.num_epoch)),
                position=self.tqdm_settings.position,
                desc=self.tqdm_settings.desc,
                leave=False,
            )
        # train
        has_ckpt = terminate = False
        if self.is_local_rank_0 and self.epoch_tqdm is None:
            print("entered training loop")
        if self.is_local_rank_0 and config_export_file is not None:
            config_export_path = os.path.join(self.workspace, config_export_file)
            with open(config_export_path, "w") as f:
                json.dump(self.export_config, f)
        for callback in self.callbacks:
            callback.before_loop(self)
        while self.state.should_train:
            try:
                self.state.epoch += 1
                if isinstance(self.train_loader, TorchDataLoader):
                    sampler = self.train_loader.loader.sampler
                    if isinstance(sampler, DistributedSampler):
                        sampler.set_epoch(self.state.epoch)
                        if isinstance(self.valid_loader, TorchDataLoader):
                            valid_sampler = self.valid_loader.loader.sampler
                            if isinstance(valid_sampler, DistributedSampler):
                                valid_sampler.set_epoch(self.state.epoch)
                step_iterator = TensorBatcher(self.train_loader, self.device)
                if self.is_local_rank_0 and self.tqdm_settings.use_step_tqdm:
                    step_tqdm = step_iterator = tqdm(
                        step_iterator,
                        total=len(self.train_loader),
                        position=self.tqdm_settings.position + 1,
                        leave=False,
                    )
                for i, batch in enumerate(step_iterator):
                    self.state.step += 1
                    step_outputs = self._step(i, batch)
                    for callback in self.callbacks:
                        callback.after_step(step_outputs, self.state)
                    monitor_results = self._monitor_step(step_outputs)
                    for callback in self.callbacks:
                        callback.after_monitor(monitor_results, self.state)
                    if self.is_local_rank_0 and monitor_results.save_checkpoint:
                        metric_outputs = monitor_results.metric_outputs
                        assert metric_outputs is not None
                        self.save_checkpoint(metric_outputs.final_score)
                    terminate = monitor_results.terminate or self.state.should_terminate
                    if terminate:
                        break
            except KeyboardInterrupt:
                if dist.is_initialized():
                    raise
                print("keyboard interrupted")
                terminate = True
            if terminate:
                break
            if self.epoch_tqdm is not None:
                self.epoch_tqdm.total = self.state.num_epoch
                self.epoch_tqdm.update()
        if self.epoch_tqdm is not None:
            if step_tqdm is not None:
                step_tqdm.close()
            self.epoch_tqdm.close()
        # restore
        if self.is_local_rank_0 and self.has_checkpoint_folder:
            if not self.tqdm_settings.in_distributed:
                print("rolling back to the best checkpoint")
            has_ckpt = self.restore_checkpoint()
        # finalize
        self.state.set_terminate()
        if self.is_local_rank_0:
            self.final_results = self._get_metrics(portion=self.config.valid_portion)
            self._logging_step(self.final_results)
            if not has_ckpt:
                self.save_checkpoint(self.final_results.final_score)
        for callback in self.callbacks:
            callback.finalize(self)
        return self

    # checkpointing

    def save_checkpoint(
        self,
        score: float,
        folder: Optional[str] = None,
        *,
        no_history: bool = False,
    ) -> None:
        if not self.is_local_rank_0:
            msg = "`save_checkpoint` should not be called when not `is_local_rank_0`"
            raise ValueError(msg)
        if folder is None:
            if self.checkpoint_folder is None:
                msg = "either `folder` or `checkpoint_folder` should be provided"
                raise ValueError(msg)
            folder = self.checkpoint_folder
        state = getattr(self, "state", None)
        pt_file = f"{PT_PREFIX}{-1 if state is None else state.step}.pt"
        if state is None:
            print(
                "`state` is not initialized, "
                "latest model will be saved and the recorded score will always be 0"
            )
            torch.save(self.model.state_dict(), os.path.join(folder, pt_file))
            with open(os.path.join(folder, SCORES_FILE), "w") as f:
                json.dump({pt_file: 0.0}, f)
            return
        # leave top_k snapshots only
        if state.max_snapshot_file > 0:
            checkpoints = get_sorted_checkpoints(folder)
            if len(checkpoints) >= state.max_snapshot_file:
                for file in checkpoints[state.max_snapshot_file - 1 :]:
                    self.checkpoint_scores.pop(file)
                    os.remove(os.path.join(folder, file))
        # pt
        torch.save(self.model.state_dict(), os.path.join(folder, pt_file))
        # scores
        scores = {} if no_history else self.checkpoint_scores
        scores[pt_file] = score
        with open(os.path.join(folder, SCORES_FILE), "w") as f:
            json.dump(sort_dict_by_value(scores, reverse=True), f)

    def restore_checkpoint(
        self,
        folder: Optional[str] = None,
        strict: bool = True,
        state_dict_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> bool:
        if not self.is_local_rank_0:
            msg = "`restore_checkpoint` should not be called when not `is_local_rank_0`"
            raise ValueError(msg)
        if folder is None:
            if self.checkpoint_folder is None:
                msg = "either `folder` or `checkpoint_folder` should be provided"
                raise ValueError(msg)
            folder = self.checkpoint_folder
        checkpoints = get_sorted_checkpoints(folder)
        if not checkpoints:
            if not self.tqdm_settings.in_distributed:
                print(f"no model file found in {folder}")
            return False
        success = False
        for checkpoint in checkpoints:
            model_file = os.path.join(folder, checkpoint)
            if not os.path.isfile(model_file):
                continue
            if not self.tqdm_settings.in_distributed:
                print(f"restoring from {model_file}")
            states = torch.load(model_file, map_location=self.device)
            if state_dict_callback is not None:
                state_dict_callback(states)
            self.model.load_state_dict(states, strict)
            success = True
            break
        return success