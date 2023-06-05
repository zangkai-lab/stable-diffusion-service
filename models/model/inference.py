import numpy as np

from abc import ABCMeta, abstractmethod
from typing import Any, Optional, Type, Dict, NamedTuple

from tools.bases.register import WithRegister
from tools.utils.type import np_dict_type

from models.data.data_loader import IDataLoader
from models.model.train_state import TrainerState
from models.model.loss import ILoss


inferences: Dict[str, Type["IInference"]] = {}


class MetricsOutputs(NamedTuple):
    final_score: float
    metric_values: Dict[str, float]
    is_positive: Dict[str, bool]


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
