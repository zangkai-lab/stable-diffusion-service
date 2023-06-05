from models.pipeline.pipeline_base import Block
from models.model.loss import ILoss
from models.config.train_config import DLConfig


@Block.register("build_loss")
class BuildLossBlock(Block):
    loss: ILoss

    def build(self, config: DLConfig) -> None:
        loss_name = config.loss_name
        loss_config = config.loss_config or {}
        self.loss = ILoss.make(loss_name, loss_config)