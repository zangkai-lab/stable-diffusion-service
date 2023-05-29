import os
import torch
import numpy as np

from abc import ABCMeta, abstractmethod, ABC
from typing import List, Optional, Union, Type, Callable, Any, Dict, TypeVar, Tuple, NamedTuple

from .core import Pipeline
from .core import Block
from .mixins import IData
from .mixins.itrainer import get_metrics

from .block import BuildModelBlock
from .block import BuildInferenceBlock
from .block import IDataLoader
from .block import SerializeDataBlock
from .block import SerializeModelBlock
from .block import MetricsOutputs
from .block import BuildLossBlock
from .block import BuildMetricsBlock

from .mixins import DLConfig
from .utils import safe_execute
from .utils import shallow_copy_dict
from .utils import is_local_rank_0
from .utils import sigmoid
from .utils import softmax


PREDICTIONS_KEY = "predictions"

np_dict_type = Dict[str, Union[np.ndarray, Any]]
tensor_dict_type = Dict[str, Union[torch.Tensor, Any]]
sample_weights_type = Optional[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]

TInferPipeline = TypeVar("TInferPipeline", bound="DLInferencePipeline", covariant=True)


class _DeviceMixin:
    build_model: BuildModelBlock

    @property
    def device(self) -> torch.device:
        return self.build_model.model.device


class _InferenceMixin:
    focuses: List[Type[Block]]
    is_built: bool

    data: Optional[IData]
    get_block: Callable[[Type[Block]], Any]
    try_get_block: Callable[[Type[Block]], Any]

    # optional callbacks

    def predict_callback(self, results: np_dict_type) -> np_dict_type:
        """changes can happen inplace"""
        return results

    # api

    @property
    def build_model(self) -> BuildModelBlock:
        return self.get_block(BuildModelBlock)

    @property
    def build_inference(self) -> BuildInferenceBlock:
        return self.get_block(BuildInferenceBlock)

    @property
    def serialize_data(self) -> SerializeDataBlock:
        return self.get_block(SerializeDataBlock)

    @property
    def serialize_model(self) -> Optional[SerializeModelBlock]:
        return self.try_get_block(SerializeModelBlock)

    @classmethod
    def build_with(  # type: ignore
        cls: Type[TInferPipeline],
        config: DLConfig,
        states: Optional[tensor_dict_type] = None,
        *,
        data: Optional[IData] = None,
    ) -> TInferPipeline:
        self = cls.init(config)
        # last focus will be the serialization block
        self.build(*[Block.make(b.__identifier__, {}) for b in cls.focuses])
        if states is not None:
            self.build_model.model.load_state_dict(states)
        self.serialize_model.verbose = False
        self.serialize_data.data = self.data = data
        self.is_built = True
        return self

    def to(  # type: ignore
        self: TInferPipeline,
        device: Union[int, str, torch.device],
    ) -> TInferPipeline:
        self.build_model.model.to(device)
        return self

    def predict(
        self,
        loader: IDataLoader,
        *,
        return_classes: bool = False,
        binary_threshold: float = 0.5,
        return_probabilities: bool = False,
        recover_labels: Optional[bool] = None,
        **kwargs: Any,
    ) -> np_dict_type:
        if not self.is_built:
            raise ValueError(
                f"`{self.__class__.__name__}` should be built beforehand, please use "
                "`DLPipelineSerializer.load_inference/evaluation` or `build_with` "
                "to get a built one!"
            )
        kw = shallow_copy_dict(kwargs)
        kw["loader"] = loader
        outputs = safe_execute(self.build_inference.inference.get_outputs, kw)
        results = outputs.forward_results
        # handle predict flags
        if return_classes and return_probabilities:
            raise ValueError(
                "`return_classes` & `return_probabilities`"
                "should not be True at the same time"
            )
        elif not return_classes and not return_probabilities:
            pass
        else:
            predictions = results[PREDICTIONS_KEY]
            if predictions.shape[1] > 2 and return_classes:
                results[PREDICTIONS_KEY] = predictions.argmax(1, keepdims=True)
            else:
                if predictions.shape[1] == 2:
                    probabilities = softmax(predictions)
                else:
                    pos = sigmoid(predictions)
                    probabilities = np.hstack([1.0 - pos, pos])
                if return_probabilities:
                    results[PREDICTIONS_KEY] = probabilities
                else:
                    classes = (probabilities[..., [1]] >= binary_threshold).astype(int)
                    results[PREDICTIONS_KEY] = classes
        # handle recover labels
        if recover_labels is None:
            recover_labels = self.data is not None
        if recover_labels:
            if self.data is None:
                msg = "`recover_labels` is set to `True` but `data` is not provided"
                raise ValueError(msg)
            y = results[PREDICTIONS_KEY]
            results[PREDICTIONS_KEY] = self.data.recover_labels(y)
        # optional callback
        results = self.predict_callback(results)
        # return
        return results


class IEvaluationPipeline(ABC):
    @abstractmethod
    def evaluate(self, loader: IDataLoader) -> MetricsOutputs:
        pass


class _EvaluationMixin(_InferenceMixin, IEvaluationPipeline):
    config: DLConfig

    @property
    def build_loss(self) -> BuildLossBlock:
        return self.get_block(BuildLossBlock)

    @property
    def build_metrics(self) -> BuildMetricsBlock:
        return self.get_block(BuildMetricsBlock)

    def evaluate(self, loader: IDataLoader, **kwargs: Any) -> MetricsOutputs:
        return get_metrics(
            self.config,
            self.build_model.model,
            self.build_loss.loss,
            self.build_metrics.metrics,
            self.build_inference.inference,
            loader,
            forward_kwargs=kwargs,
        )


class TrainingPipeline(
    Pipeline,
    _DeviceMixin,
    _EvaluationMixin,
    metaclass=ABCMeta,
):
    is_built = False

    @property
    @abstractmethod
    def set_defaults_block(self) -> Block:
        pass

    @property
    @abstractmethod
    def set_trainer_defaults_block(self) -> Block:
        pass

    @property
    def build_trainer(self) -> BuildTrainerBlock:
        return self.get_block(BuildTrainerBlock)

    @property
    def building_blocks(self) -> List[Block]:
        return [
            self.set_defaults_block,
            PrepareWorkplaceBlock(),
            ExtractStateInfoBlock(),
            BuildLossBlock(),
            BuildModelBlock(),
            BuildMetricsBlock(),
            BuildInferenceBlock(),
            self.set_trainer_defaults_block,
            BuildMonitorsBlock(),
            BuildCallbacksBlock(),
            BuildOptimizersBlock(),
            BuildTrainerBlock(),
            RecordNumSamplesBlock(),
            ReportBlock(),
            TrainingBlock(),
            SerializeDataBlock(),
            SerializeModelBlock(),
            SerializeOptimizerBlock(),
        ]

    def after_load(self) -> None:
        self.is_built = True
        workspace = prepare_workspace_from("_logs")
        self.config.workspace = workspace

    def prepare(self, data: IData, sample_weights: sample_weights_type = None) -> None:
        self.data = data.set_sample_weights(sample_weights)
        self.training_workspace = self.config.workspace
        if not self.is_built:
            self.build(*self.building_blocks)
            self.is_built = True
        else:
            for block in self.blocks:
                block.training_workspace = self.training_workspace

    def fit(
        self,
        data: IData,
        *,
        sample_weights: sample_weights_type = None,
        cuda: Optional[Union[int, str]] = None,
    ) -> "TrainingPipeline":
        # build pipeline
        self.prepare(data, sample_weights)
        # check rank 0
        workspace = self.config.workspace if is_local_rank_0() else None
        # save data info
        if workspace is not None:
            Serializer.save(
                os.path.join(workspace, SerializeDataBlock.package_folder),
                data,
                save_npd=False,
            )
        # run pipeline
        self.run(data, cuda=cuda)
        # save pipeline
        if workspace is not None:
            pipeline_folder = DLPipelineSerializer.pipeline_folder
            DLPipelineSerializer.save(self, os.path.join(workspace, pipeline_folder))
        # return
        return self