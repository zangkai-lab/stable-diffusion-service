from typing import NamedTuple, Dict, Any, Tuple, Optional, Union, List, Type
from abc import ABCMeta, abstractmethod

from tools.bases.register import WithRegister
from tools.utils.type import np_dict_type, configs_type

from models.data.data_loader import IDataLoader
from models.model.constant import LABEL_KEY, PREDICTIONS_KEY
from models.model.mode_context import eval_context
from models.model.loss import ILoss
from models.model.model_dl import IDLModel
from models.model.inference import IInference
from models.model.train_state import TrainerState
from models.config.train_config import TrainerConfig


metric_dict: Dict[str, Type["IMetric"]] = {}


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