from abc import abstractmethod
from abc import ABC

from models.data.data_loader import IDataLoader
from models.model.metrics import MetricsOutputs
from models.pipeline.pipeline_base import Pipeline, PipelineTypes
from models.pipeline.pipeline_inference import DLInferencePipeline
from models.pipeline.mixin.mixin_evaluation import EvaluationMixin
from models.pipeline.block.block_loss import BuildLossBlock
from models.pipeline.block.block_model import BuildModelBlock
from models.pipeline.block.block_metrics import BuildMetricsBlock
from models.pipeline.block.block_inference import BuildInferenceBlock
from models.pipeline.block.block_serialize import SerializeDataBlock
from models.pipeline.block.block_serialize import SerializeModelBlock


class IEvaluationPipeline(ABC):
    @abstractmethod
    def evaluate(self, loader: IDataLoader) -> MetricsOutputs:
        pass


@Pipeline.register(PipelineTypes.DL_EVALUATION)
class DLEvaluationPipeline(DLInferencePipeline, EvaluationMixin):
    focuses = [
        BuildLossBlock,
        BuildModelBlock,
        BuildMetricsBlock,
        BuildInferenceBlock,
        SerializeDataBlock,
        SerializeModelBlock,
    ]