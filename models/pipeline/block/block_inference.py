from typing import List, Type

from models.pipeline.pipeline_base import Block
from models.config.train_config import DLConfig
from models.pipeline.block.block_model import BuildModelBlock
from models.model.inference import IInference


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