from typing import Any

from models.config.train_config import DLConfig
from models.data.data_loader import IDataLoader
from models.model.metrics import MetricsOutputs, get_metrics
from models.pipeline.mixin.mixin_inference import InferenceMixin
from models.pipeline.pipeline_evaluation import IEvaluationPipeline
from models.pipeline.block.block_loss import BuildLossBlock
from models.pipeline.block.block_metrics import BuildMetricsBlock


class EvaluationMixin(InferenceMixin, IEvaluationPipeline):
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