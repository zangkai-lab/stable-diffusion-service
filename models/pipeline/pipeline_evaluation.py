from abc import abstractmethod
from abc import ABC

from models.data.data_loader import IDataLoader
from models.model.metrics import MetricsOutputs


class IEvaluationPipeline(ABC):
    @abstractmethod
    def evaluate(self, loader: IDataLoader) -> MetricsOutputs:
        pass
