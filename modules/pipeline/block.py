import os
import torch
import inspect
import math
import shutil
import json

import torch.nn as nn
import numpy as np

from torch import Tensor
from copy import deepcopy
from typing import OrderedDict as OrderedDictType
from typing import Any, Optional, Tuple, Union, Dict, List, Callable, Type, Protocol, TypeVar
from typing import NamedTuple
from typing import ContextManager
from dataclasses import dataclass
from abc import ABCMeta, abstractmethod, ABC
from collections import OrderedDict

from .core import Block
from .mixins import WithRegister
from .mixins import Config
from .mixins import DLConfig
from .mixins import DataClassBase
from .mixins import IData
from .mixins import ISerializable, ISerializableArrays, TSerializable
from .mixins import TrainerConfig
from .mixins import PrecisionType
from .mixins import ITrainer

from .utils import shallow_copy_dict
from .utils import fix_denormal_states
from .utils import safe_execute
from .utils import get_clones
from .utils import get_world_size

from models.utils import get_device
from ..apis.utils import sort_dict_by_value


try:
    import onnx
except:
    onnx = None
try:
    from onnxsim import simplify as onnx_simplify


    def get_inputs(model: onnx.ModelProto) -> List[onnx.ValueInfoProto]:
        initializer_names = [x.name for x in model.graph.initializer]
        return [inp for inp in model.graph.input if inp.name not in initializer_names]


    def get_input_names(model: onnx.ModelProto) -> List[str]:
        input_names = [inp.name for inp in get_inputs(model)]
        return input_names

except:
    onnx_simplify = get_input_names = None  # type: ignore

LOSS_KEY = "loss"
INPUT_KEY = "input"
LATENT_KEY = "latent"
PREDICTIONS_KEY = "predictions"
LABEL_KEY = "labels"

PT_PREFIX = "model_"
SCORES_FILE = "scores.json"
CHECKPOINTS_FOLDER = "checkpoints"

TLoss = TypeVar("TLoss", bound="ILoss", covariant=True)

model_dict: Dict[str, Type["IDLModel"]] = {}
np_dict_type = Dict[str, Union[np.ndarray, Any]]
tensor_dict_type = Dict[str, Union[torch.Tensor, Any]]
forward_results_type = Union[Tensor, tensor_dict_type]
loss_dict: Dict[str, Type["ILoss"]] = {}
losses_type = Union[Tensor, tensor_dict_type]
metric_dict: Dict[str, Type["IMetric"]] = {}
configs_type = Optional[Union[List[Dict[str, Any]], Dict[str, Any]]]


def get_requirements(fn: Any) -> List[str]:
    if isinstance(fn, type):
        fn = fn.__init__  # type: ignore
    requirements = []
    signature = inspect.signature(fn)
    for k, param in signature.parameters.items():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        if param.default is not inspect.Parameter.empty:
            continue
        requirements.append(k)
    return requirements


class InjectDefaultsMixin:
    _defaults: OrderedDictType

    def __init__(self) -> None:
        self._defaults = OrderedDict()

    def process_defaults(self, _defaults: OrderedDictType) -> None:
        for k, v in self._defaults.items():
            _defaults[k] = v


class IWithRequirements:
    @classmethod
    def requirements(cls) -> List[str]:
        requirements = get_requirements(cls)
        requirements.remove("self")
        return requirements


class mode_context:
    """
    Help entering specific mode and recovering previous mode

    This is a context controller for entering specific mode at the beginning
    and back to previous mode at the end.

    """

    def __init__(
            self,
            module: nn.Module,
            *,
            to_train: Optional[bool],
            use_grad: Optional[bool],
            use_inference: Optional[bool] = None,
    ):
        self._to_train = to_train
        self._module, self._training = module, module.training
        self._cache = {p: p.requires_grad for p in module.parameters()}
        if use_grad is not None:
            for p in module.parameters():
                p.requires_grad_(use_grad)
        if use_grad is None:
            self._grad_context: Optional[ContextManager] = None
        else:
            self._grad_context = torch.enable_grad() if use_grad else torch.no_grad()
        if use_inference is None:
            self._inference_context: Optional[ContextManager] = None
        else:
            self._inference_context = torch.inference_mode(use_inference)

    def __enter__(self) -> None:
        if self._to_train is not None:
            self._module.train(mode=self._to_train)
        if self._grad_context is not None:
            self._grad_context.__enter__()
        if self._inference_context is not None:
            self._inference_context.__enter__()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._to_train is not None:
            self._module.train(mode=self._training)
        if self._inference_context is not None:
            self._inference_context.__exit__(exc_type, exc_val, exc_tb)
        if self._grad_context is not None:
            self._grad_context.__exit__(exc_type, exc_val, exc_tb)
        for p, v in self._cache.items():
            if p.requires_grad != v:
                p.requires_grad_(v)


class eval_context(mode_context):
    """
    Useful when we need to predict something with our PyTorch models during training.
    """

    def __init__(
            self,
            module: nn.Module,
            *,
            use_grad: Optional[bool] = False,
            use_inference: Optional[bool] = None,
    ):
        if use_inference is None and use_grad is not None:
            use_inference = not use_grad
        super().__init__(
            module,
            to_train=False,
            use_grad=use_grad,
            use_inference=use_inference,
        )


class IDLModel(
    nn.Module,
    WithRegister["IDLModel"],
    IWithRequirements,
    metaclass=ABCMeta,
):
    d = model_dict

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()

    # optional callbacks

    def init_with_trainer(self, trainer: "ITrainer") -> None:
        pass

    def permute_trainer_config(self, trainer_config: "TrainerConfig") -> None:
        pass

    def get_forward_args(
            self,
            batch_idx: int,
            batch: tensor_dict_type,
            state: Optional["TrainerState"] = None,
            **kwargs: Any,
    ) -> Tuple[Any, ...]:
        return (batch[INPUT_KEY],)

    def postprocess(
            self,
            batch_idx: int,
            batch: tensor_dict_type,
            forward_results: forward_results_type,
            state: Optional["TrainerState"] = None,
            **kwargs: Any,
    ) -> tensor_dict_type:
        if isinstance(forward_results, dict):
            return forward_results
        if isinstance(forward_results, Tensor):
            return {PREDICTIONS_KEY: forward_results}
        raise ValueError(f"unrecognized forward results occurred: {forward_results}")

    # api

    def run(
            self,
            batch_idx: int,
            batch: tensor_dict_type,
            state: Optional["TrainerState"] = None,
            **kwargs: Any,
    ) -> tensor_dict_type:
        args = self.get_forward_args(batch_idx, batch, state, **kwargs)
        forward_results = self(*args)
        outputs = self.postprocess(batch_idx, batch, forward_results, state, **kwargs)
        return outputs

    def onnx_forward(self, batch: tensor_dict_type) -> Any:
        return self.run(0, batch)

    def summary_forward(self, batch: tensor_dict_type) -> None:
        self.onnx_forward(batch)

    def to_onnx(
            self,
            export_file: str,
            input_sample: tensor_dict_type,
            dynamic_axes: Optional[Union[List[int], Dict[int, str]]] = None,
            *,
            opset: int = 11,
            simplify: bool = True,
            forward_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
            output_names: Optional[List[str]] = None,
            num_samples: Optional[int] = None,
            verbose: bool = True,
            **kwargs: Any,
    ) -> "IDLModel":
        # prepare
        device = get_device(self)
        model = self.cpu()
        if num_samples is not None:
            input_sample = {k: v[:num_samples] for k, v in input_sample.items()}
        onnx_forward = forward_fn or model.onnx_forward
        input_names = sorted(input_sample.keys())
        if output_names is None:
            if forward_fn is not None:
                msg = "`output_names` should be provided when `forward_fn` is provided"
                raise ValueError(msg)
            with eval_context(model):
                forward_results = onnx_forward(shallow_copy_dict(input_sample))
            if not isinstance(forward_results, dict):
                forward_results = {PREDICTIONS_KEY: forward_results}
            output_names = sorted(forward_results.keys())
        # setup
        kwargs = shallow_copy_dict(kwargs)
        kwargs["input_names"] = input_names
        kwargs["output_names"] = output_names
        kwargs["opset_version"] = opset
        kwargs["export_params"] = True
        kwargs["do_constant_folding"] = True
        if dynamic_axes is None:
            dynamic_axes = {}
        elif isinstance(dynamic_axes, list):
            dynamic_axes = {axis: f"axis.{axis}" for axis in dynamic_axes}
        if num_samples is None:
            dynamic_axes[0] = "batch_size"
        dynamic_axes_settings = {}
        for name in input_names + output_names:
            dynamic_axes_settings[name] = dynamic_axes
        kwargs["dynamic_axes"] = dynamic_axes_settings
        kwargs["verbose"] = verbose

        # export

        class ONNXWrapper(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = model

            def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
                rs = onnx_forward(batch)
                if isinstance(rs, Tensor):
                    return {k: rs for k in output_names}  # type: ignore
                return {k: rs[k] for k in output_names}  # type: ignore

        m_onnx = ONNXWrapper()
        original_states = model.state_dict()
        fixed_states = fix_denormal_states(original_states, verbose=verbose)
        with eval_context(m_onnx):
            model.load_state_dict(fixed_states)
            torch.onnx.export(
                m_onnx,
                ({k: input_sample[k] for k in input_names}, {}),
                export_file,
                **shallow_copy_dict(kwargs),
            )
            model.load_state_dict(original_states)
            if not simplify:
                return self.to(device)
            if onnx is None:
                print(
                    "`onnx` is not installed, "
                    "so the exported onnx models will not be simplified"
                )
                return self.to(device)
            if onnx_simplify is None or get_input_names is None:
                print(
                    "`onnx-simplifier` is not installed, "
                    "so the exported onnx models will not be simplified"
                )
                return self.to(device)
            try:
                onnx_model = onnx.load(export_file)
                final_input_names = get_input_names(onnx_model)
                model_simplified, check = onnx_simplify(
                    onnx_model,
                    test_input_shapes={
                        name: tensor.shape
                        for name, tensor in input_sample.items()
                        if name in final_input_names
                    },
                )
            except Exception as err:
                if verbose:
                    print(f"Failed to simplify ONNX models ({err})")
                model_simplified = None
                check = False
            if verbose:
                tag = " " if check else " not "
                print(f"Simplified ONNX models is{tag}validated!")
            if check and model_simplified is not None:
                onnx.save(model_simplified, export_file)
        return self.to(device)


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


class EnsembleFn(Protocol):
    def __call__(self, key: str, tensors: List[Tensor]) -> Tensor:
        pass


class DLEnsembleModel(nn.Module):
    ensemble_fn: Optional[EnsembleFn]

    def __init__(self, m: IDLModel, num_repeat: int) -> None:
        super().__init__()
        self.ms = get_clones(m, num_repeat)
        self.ensemble_fn = None

    def forward(self, *args: Any) -> forward_results_type:
        outputs: Dict[str, List[Tensor]] = {}
        for m in self.ms:
            m_outputs = m(*args)
            if isinstance(m_outputs, Tensor):
                m_outputs = {PREDICTIONS_KEY: m_outputs}
            for k, v in m_outputs.items():
                outputs.setdefault(k, []).append(v)
        final_results: tensor_dict_type = {}
        for k in sorted(outputs):
            if self.ensemble_fn is None:
                v = torch.stack(outputs[k]).mean(0)
            else:
                v = safe_execute(self.ensemble_fn, dict(key=k, tensors=outputs[k]))
            final_results[k] = v
        return final_results

    def run(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional["TrainerState"] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        m: IDLModel = self.ms[0]
        args = m.get_forward_args(batch_idx, batch, state, **kwargs)
        forward_results = self(*args)
        outputs = m.postprocess(batch_idx, batch, forward_results, state, **kwargs)
        return outputs


@Block.register("build_model")
class BuildModelBlock(InjectDefaultsMixin, Block):
    model: IDLModel

    def build(self, config: Union[DLConfig, MLConfig]) -> None:
        model_name = config.model_name
        model_config = config.model_config or {}
        if isinstance(config, MLConfig):
            self._setup_ml_model(config, model_config)
        num_repeat = config.num_repeat
        m = safe_execute(IDLModel.get(model_name), model_config)
        if num_repeat is None:
            self.model = m
        else:
            self.model = DLEnsembleModel(m, num_repeat)

    def _setup_ml_model(self, config: MLConfig, model_config: Dict[str, Any]) -> None:
        if config.encoder_settings is None or config.index_mapping is None:
            encoder_settings = config.encoder_settings
        else:
            encoder_settings = {}
            for k, v in config.encoder_settings.items():
                encoder_settings[str(config.index_mapping[k])] = v
        model_config["encoder_settings"] = encoder_settings
        model_config["global_encoder_settings"] = config.global_encoder_settings
        mc = self._defaults.setdefault("model_config", {})
        if encoder_settings is not None:
            d = {k: v.asdict() for k, v in encoder_settings.items()}
            mc["encoder_settings"] = d
        if config.global_encoder_settings is not None:
            ges = config.global_encoder_settings.asdict()
            self._defaults["global_encoder_settings"] = ges
        self._defaults["index_mapping"] = config.index_mapping


inferences: Dict[str, Type["IInference"]] = {}


class IDataset(ABC):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, item: Union[int, List[int], np.ndarray]) -> Dict[str, Any]:
        pass


class context_error_handler:
    """Util class which provides exception handling when using context manager."""

    @property
    def exception_suffix(self):
        return ""

    def _normal_exit(self, exc_type, exc_val, exc_tb):
        pass

    def _exception_exit(self, exc_type, exc_val, exc_tb):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_type:
            self._normal_exit(exc_type, exc_val, exc_tb)
        else:
            self._exception_exit(exc_type, exc_val, exc_tb)


class IDataLoader(ABC):
    dataset: IDataset
    batch_size: int

    def __init__(self, *, sample_weights: Optional[np.ndarray] = None):
        self.sample_weights = sample_weights

    @abstractmethod
    def __iter__(self) -> "IDataLoader":
        pass

    @abstractmethod
    def __next__(self) -> np_dict_type:
        pass

    @abstractmethod
    def disable_shuffle(self) -> None:
        pass

    @abstractmethod
    def recover_shuffle(self) -> None:
        pass

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)

    def copy(self) -> "IDataLoader":
        return deepcopy(self)

    def temporarily_disable_shuffle(self) -> context_error_handler:
        class _(context_error_handler):
            def __init__(self, loader: IDataLoader):
                self.loader = loader

            def __enter__(self) -> None:
                self.loader.disable_shuffle()

            def _normal_exit(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                self.loader.recover_shuffle()

        return _(self)

    def get_full_batch(self) -> np_dict_type:
        batch_size = self.batch_size
        self.batch_size = len(self.dataset)
        full_batch = next(iter(self))
        self.batch_size = batch_size
        return full_batch


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


class ILoss(nn.Module, WithRegister[TLoss], metaclass=ABCMeta):
    d = loss_dict
    placeholder_key = "[PLACEHOLDER]"

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def _reduce(self, losses: Tensor) -> Tensor:
        if self.reduction == "none":
            return losses
        if self.reduction == "mean":
            return losses.mean()
        if self.reduction == "sum":
            return losses.sum()
        raise NotImplementedError(f"reduction '{self.reduction}' is not implemented")

    # optional callbacks

    def get_forward_args(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
    ) -> Tuple[Any, ...]:
        return forward_results[PREDICTIONS_KEY], batch[LABEL_KEY]

    def postprocess(
        self,
        losses: losses_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
    ) -> tensor_dict_type:
        if not isinstance(losses, dict):
            losses = {LOSS_KEY: losses}
        return {k: self._reduce(v) for k, v in losses.items()}

    # api

    def run(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
    ) -> tensor_dict_type:
        args = self.get_forward_args(forward_results, batch, state)
        losses = self(*args)
        losses = self.postprocess(losses, batch, state)
        return losses


class MetricsOutputs(NamedTuple):
    final_score: float
    metric_values: Dict[str, float]
    is_positive: Dict[str, bool]


class IMetric(WithRegister["IMetric"], metaclass=ABCMeta):
    d = metric_dict

    def __init__(self, *args: Any, **kwargs: Any):
        pass

    # abstract

    @property
    @abstractmethod
    def is_positive(self) -> bool:
        pass

    @abstractmethod
    def forward(self, *args: Any) -> float:
        pass

    # optional callback

    @property
    def requires_all(self) -> bool:
        """
        Specify whether this Metric needs 'all' data.

        Typical metrics often does not need to evaluate itself on the entire dataset,
        but some does need to avoid corner cases. (for instance, the AUC metrics may
        fail to evaluate itself on only a batch, because the labels in this batch may
        be all the same, which breaks the calculation of AUC).
        """
        return False

    def get_forward_args(
        self,
        np_batch: np_dict_type,
        np_outputs: np_dict_type,
        loader: Optional[IDataLoader] = None,
    ) -> Tuple[Any, ...]:
        return np_outputs[PREDICTIONS_KEY], np_batch[LABEL_KEY]

    # api

    def run(
        self,
        np_batch: np_dict_type,
        np_outputs: np_dict_type,
        loader: Optional[IDataLoader] = None,
    ) -> float:
        args = self.get_forward_args(np_batch, np_outputs, loader)
        return self.forward(*args)

    @classmethod
    def fuse(
        cls,
        names: Union[str, List[str]],
        configs: configs_type = None,
        *,
        metric_weights: Optional[Dict[str, float]] = None,
    ) -> "IMetric":
        metrics = IMetric.make_multiple(names, configs)
        if isinstance(metrics, IMetric):
            return metrics
        return MultipleMetrics(metrics, weights=metric_weights)

    def evaluate(
        self,
        np_batch: np_dict_type,
        np_outputs: np_dict_type,
        loader: Optional[IDataLoader] = None,
    ) -> MetricsOutputs:
        metric = self.run(np_batch, np_outputs, loader)
        score = metric * (1.0 if self.is_positive else -1.0)
        k = self.__identifier__
        return MetricsOutputs(score, {k: metric}, {k: self.is_positive})


class MultipleMetrics(IMetric):
    @property
    def is_positive(self) -> bool:
        raise NotImplementedError

    @property
    def requires_all(self) -> bool:
        return any(metric.requires_all for metric in self.metrics)

    def forward(self, *args: Any) -> float:
        raise NotImplementedError

    def __init__(
        self,
        metric_list: List[IMetric],
        *,
        weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.metrics = metric_list
        self.weights = weights or {}
        self.__identifier__ = " | ".join(m.__identifier__ for m in metric_list)

    def evaluate(
        self,
        np_batch: np_dict_type,
        np_outputs: np_dict_type,
        loader: Optional[IDataLoader] = None,
    ) -> MetricsOutputs:
        scores: List[float] = []
        weights: List[float] = []
        metrics_values: Dict[str, float] = {}
        is_positive: Dict[str, bool] = {}
        for metric in self.metrics:
            metric_outputs = metric.evaluate(np_batch, np_outputs, loader)
            w = self.weights.get(metric.__identifier__, 1.0)
            weights.append(w)
            scores.append(metric_outputs.final_score * w)
            metrics_values.update(metric_outputs.metric_values)
            is_positive.update(metric_outputs.is_positive)
        return MetricsOutputs(sum(scores) / sum(weights), metrics_values, is_positive)


class InferenceOutputs(NamedTuple):
    forward_results: np_dict_type
    labels: Optional[np.ndarray]
    metric_outputs: Optional[MetricsOutputs]
    loss_items: Optional[Dict[str, float]]


class IInference(WithRegister["IInference"], metaclass=ABCMeta):
    d = inferences
    use_grad_in_predict = False

    @abstractmethod
    def get_outputs(
        self,
        loader: IDataLoader,
        *,
        portion: float = 1.0,
        state: Optional[TrainerState] = None,
        metrics: Optional["IMetric"] = None,
        loss: Optional[ILoss] = None,
        return_outputs: bool = True,
        stack_outputs: bool = True,
        use_tqdm: bool = False,
        **kwargs: Any,
    ) -> InferenceOutputs:
        pass


@Block.register("build_inference")
class BuildInferenceBlock(Block):
    inference: IInference

    def build(self, config: DLConfig) -> None:
        inference_type = config.inference_type
        inference_kw = dict(model=self.build_model.model)
        self.inference = IInference.make(inference_type, inference_kw)

    @property
    def requirements(self) -> List[Type[Block]]:
        return [BuildModelBlock]

    @property
    def build_model(self) -> BuildModelBlock:
        return self.get_previous(BuildModelBlock)


class Serializer:
    id_file: str = "id.txt"
    info_file: str = "info.json"
    npd_folder: str = "npd"

    @classmethod
    def save_info(
        cls,
        folder: str,
        *,
        info: Optional[Dict[str, Any]] = None,
        serializable: Optional[ISerializable] = None,
    ) -> None:
        os.makedirs(folder, exist_ok=True)
        if info is None and serializable is None:
            raise ValueError("either `info` or `serializable` should be provided")
        if info is None:
            info = serializable.to_info()
        with open(os.path.join(folder, cls.info_file), "w") as f:
            json.dump(info, f)

    @classmethod
    def load_info(cls, folder: str) -> Dict[str, Any]:
        return cls.try_load_info(folder, strict=True)

    @classmethod
    def try_load_info(
        cls,
        folder: str,
        *,
        strict: bool = False,
    ) -> Optional[Dict[str, Any]]:
        info_path = os.path.join(folder, cls.info_file)
        if not os.path.isfile(info_path):
            if not strict:
                return
            raise ValueError(f"'{info_path}' does not exist")
        with open(info_path, "r") as f:
            info = json.load(f)
        return info

    @classmethod
    def save_npd(
        cls,
        folder: str,
        *,
        npd: Optional[np_dict_type] = None,
        serializable: Optional[ISerializableArrays] = None,
    ) -> None:
        os.makedirs(folder, exist_ok=True)
        if npd is None and serializable is None:
            raise ValueError("either `npd` or `serializable` should be provided")
        if npd is None:
            npd = serializable.to_npd()
        npd_folder = os.path.join(folder, cls.npd_folder)
        os.makedirs(npd_folder, exist_ok=True)
        for k, v in npd.items():
            np.save(os.path.join(npd_folder, f"{k}.npy"), v)

    @classmethod
    def load_npd(cls, folder: str) -> np_dict_type:
        os.makedirs(folder, exist_ok=True)
        npd_folder = os.path.join(folder, cls.npd_folder)
        if not os.path.isdir(npd_folder):
            raise ValueError(f"'{npd_folder}' does not exist")
        npd = {}
        for file in os.listdir(npd_folder):
            key = os.path.splitext(file)[0]
            npd[key] = np.load(os.path.join(npd_folder, file))
        return npd

    @classmethod
    def save(
        cls,
        folder: str,
        serializable: ISerializable,
        *,
        save_npd: bool = True,
    ) -> None:
        cls.save_info(folder, serializable=serializable)
        if save_npd and isinstance(serializable, ISerializableArrays):
            cls.save_npd(folder, serializable=serializable)
        with open(os.path.join(folder, cls.id_file), "w") as f:
            f.write(serializable.__identifier__)

    @classmethod
    def load(
        cls,
        folder: str,
        base: Type[TSerializable],
        *,
        swap_id: Optional[str] = None,
        swap_info: Optional[Dict[str, Any]] = None,
        load_npd: bool = True,
    ) -> TSerializable:
        serializable = cls.load_empty(folder, base, swap_id=swap_id)
        serializable.from_info(swap_info or cls.load_info(folder))
        if load_npd and isinstance(serializable, ISerializableArrays):
            serializable.from_npd(cls.load_npd(folder))
        return serializable

    @classmethod
    def load_empty(
        cls,
        folder: str,
        base: Type[TSerializable],
        *,
        swap_id: Optional[str] = None,
    ) -> TSerializable:
        if swap_id is not None:
            s_type = swap_id
        else:
            id_path = os.path.join(folder, cls.id_file)
            if not os.path.isfile(id_path):
                raise ValueError(f"cannot find '{id_path}'")
            with open(id_path, "r") as f:
                s_type = f.read().strip()
        return base.make(s_type, {})


@Block.register("serialize_data")
class SerializeDataBlock(Block):
    data: Optional[IData]
    config: DLConfig
    package_folder: str = "data_module"

    def build(self, config: DLConfig) -> None:
        self.data = None
        self.config = config

    def save_extra(self, folder: str) -> None:
        if not self.is_local_rank_0:
            return
        if self.training_workspace is not None:
            data_folder = os.path.join(self.training_workspace, self.package_folder)
            shutil.copytree(data_folder, folder)
        elif self.data is not None:
            Serializer.save(folder, self.data, save_npd=False)

    def load_from(self, folder: str) -> None:
        if os.path.isdir(folder):
            self.data = Serializer.load(folder, IData, load_npd=False)


def get_scores(folder: str) -> Dict[str, float]:
    scores_path = os.path.join(folder, SCORES_FILE)
    if not os.path.isfile(scores_path):
        return {}
    with open(scores_path, "r") as f:
        return json.load(f)


def get_sorted_checkpoints(checkpoint_folder: str) -> List[str]:
    # better checkpoints will be placed earlier,
    #  which means `checkpoints[0]` is the best checkpoint
    scores = get_scores(checkpoint_folder)
    if not scores:
        return []
    return list(sort_dict_by_value(scores, reverse=True).keys())


@Block.register("serialize_model")
class SerializeModelBlock(Block):
    config: DLConfig

    verbose: bool = True
    ckpt_folder: Optional[str] = None
    ckpt_scores: Optional[Dict[str, float]] = None

    def build(self, config: DLConfig) -> None:
        self.config = config
        self.best_score = 0.0

    @property
    def requirements(self) -> List[Type[Block]]:
        return [BuildModelBlock]

    @property
    def build_model(self) -> BuildModelBlock:
        return self.get_previous(BuildModelBlock)

    def save_extra(self, folder: str) -> None:
        if not self.is_local_rank_0:
            return
        warn_msg = "no checkpoints found at {}, current models states will be saved"
        if self.training_workspace is not None:
            ckpt_folder = os.path.join(self.training_workspace, CHECKPOINTS_FOLDER)
            if get_sorted_checkpoints(ckpt_folder):
                shutil.copytree(ckpt_folder, folder)
            else:
                if self.verbose:
                    print(warn_msg.format(ckpt_folder))
                self._save_current(folder)
            return
        if self.ckpt_folder is None or self.ckpt_scores is None:
            if self.verbose:
                print("current models states will be saved")
            self._save_current(folder)
        else:
            any_saved = False
            filtered_scores = {}
            os.makedirs(folder, exist_ok=True)
            for file, score in self.ckpt_scores.items():
                ckpt_path = os.path.join(self.ckpt_folder, file)
                if not os.path.isfile(ckpt_path):
                    if self.verbose:
                        msg = f"cannot find checkpoint at '{ckpt_path}', did you delete it?"
                        print(msg)
                    continue
                any_saved = True
                filtered_scores[file] = score
                shutil.copyfile(ckpt_path, os.path.join(folder, file))
            if any_saved:
                with open(os.path.join(folder, SCORES_FILE), "w") as f:
                    json.dump(filtered_scores, f)
            else:
                if self.verbose:
                    print(warn_msg.format(self.ckpt_folder))
                self._save_current(folder)

    def load_from(self, folder: str) -> None:
        model = self.build_model.model
        best_file = get_sorted_checkpoints(folder)[0]
        model.load_state_dict(torch.load(os.path.join(folder, best_file)))
        scores = get_scores(folder)
        self.ckpt_folder = folder
        self.ckpt_scores = scores

    def _save_current(self, folder: str) -> None:
        os.makedirs(folder, exist_ok=True)
        latest_file = f"{PT_PREFIX}-1.pt"
        latest_path = os.path.join(folder, latest_file)
        new_scores_path = os.path.join(folder, SCORES_FILE)
        torch.save(self.build_model.model.state_dict(), latest_path)
        with open(new_scores_path, "w") as f:
            json.dump({latest_file: 0.0}, f)


@Block.register("build_loss")
class BuildLossBlock(Block):
    loss: ILoss

    def build(self, config: DLConfig) -> None:
        loss_name = config.loss_name
        loss_config = config.loss_config or {}
        self.loss = ILoss.make(loss_name, loss_config)


@Block.register("build_metrics")
class BuildMetricsBlock(Block):
    metrics: Optional[IMetric]

    def build(self, config: DLConfig) -> None:
        # build metrics
        metric_names = config.metric_names
        metric_configs = config.metric_configs
        metric_weights = config.metric_weights
        if metric_names is None:
            self.metrics = None
        else:
            self.metrics = IMetric.fuse(
                metric_names,
                metric_configs,
                metric_weights=metric_weights,
            )
        # check losses-as-metrics
        loss_metrics_weights = config.loss_metrics_weights
        use_losses_as_metrics = config.use_losses_as_metrics
        if self.metrics is None:
            if use_losses_as_metrics is None:
                use_losses_as_metrics = True
            if not use_losses_as_metrics:
                msg = "`metrics` should be provided when not `use_losses_as_metrics`"
                raise ValueError(msg)
        if loss_metrics_weights is not None:
            if use_losses_as_metrics is None:
                use_losses_as_metrics = True
            elif not use_losses_as_metrics:
                raise ValueError(
                    "`use_losses_as_metrics` should not be False "
                    "when `loss_metrics_weights` is provided"
                )
        config.use_losses_as_metrics = use_losses_as_metrics


@Block.register("build_trainer")
class BuildTrainerBlock(Block):
    trainer: ITrainer

    def build(self, config: DLConfig) -> None:
        self.trainer = Trainer(config)