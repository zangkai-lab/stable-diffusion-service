from models.pipeline.pipeline_base import Block
from models.config.train_config import DLConfig
from models.model.trainer import ITrainer, Trainer


@Block.register("build_trainer")
class BuildTrainerBlock(Block):
    trainer: ITrainer

    def build(self, config: DLConfig) -> None:
        self.trainer = Trainer(config)
