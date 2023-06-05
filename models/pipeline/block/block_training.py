from typing import OrderedDict as OrderedDictType
from typing import Any, List, Optional, Type, Union

from models.pipeline.pipeline_base import Block
from models.config.train_config import DLConfig
from models.data.data import IData

from models.pipeline.block.block_loss import BuildLossBlock
from models.pipeline.block.block_model import BuildModelBlock
from models.pipeline.block.block_metrics import BuildMetricsBlock
from models.pipeline.block.block_inference import BuildInferenceBlock
from models.pipeline.block.block_optimizer import BuildOptimizersBlock
from models.pipeline.block.block_monitor import BuildMonitorsBlock
from models.pipeline.block.block_callback import BuildCallbacksBlock
from models.pipeline.block.block_trainer import BuildTrainerBlock


@Block.register("training")
class TrainingBlock(Block):
    trainer_config_file = "trainer_config.json"

    def build(self, config: DLConfig) -> None:
        pass

    @property
    def requirements(self) -> List[Type[Block]]:
        return [
            BuildLossBlock,
            BuildModelBlock,
            BuildMetricsBlock,
            BuildInferenceBlock,
            BuildOptimizersBlock,
            BuildMonitorsBlock,
            BuildCallbacksBlock,
            BuildTrainerBlock,
        ]

    @property
    def build_loss(self) -> BuildLossBlock:
        return self.get_previous(BuildLossBlock)

    @property
    def build_model(self) -> BuildModelBlock:
        return self.get_previous(BuildModelBlock)

    @property
    def build_metrics(self) -> BuildMetricsBlock:
        return self.get_previous(BuildMetricsBlock)

    @property
    def build_inference(self) -> BuildInferenceBlock:
        return self.get_previous(BuildInferenceBlock)

    @property
    def build_optimizers(self) -> BuildOptimizersBlock:
        return self.get_previous(BuildOptimizersBlock)

    @property
    def build_monitors(self) -> BuildMonitorsBlock:
        return self.get_previous(BuildMonitorsBlock)

    @property
    def build_callbacks(self) -> BuildCallbacksBlock:
        return self.get_previous(BuildCallbacksBlock)

    @property
    def build_trainer(self) -> BuildTrainerBlock:
        return self.get_previous(BuildTrainerBlock)

    def run(
        self,
        data: IData,
        _defaults: OrderedDictType,
        *,
        cuda: Optional[Union[int, str]] = None,
        **kwargs: Any,
    ) -> None:
        if cuda is not None:
            cuda = str(cuda)
        self.build_trainer.trainer.fit(
            data,
            self.build_loss.loss,
            self.build_model.model,
            self.build_metrics.metrics,
            self.build_inference.inference,
            self.build_optimizers.optimizers,
            self.build_optimizers.schedulers,
            self.build_monitors.monitors,
            self.build_callbacks.callbacks,
            self.build_optimizers.schedulers_requires_metric,
            config_export_file=self.trainer_config_file,
            cuda=cuda,
        )