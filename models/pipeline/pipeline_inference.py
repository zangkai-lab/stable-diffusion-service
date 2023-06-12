from typing import TypeVar

from models.pipeline.pipeline_base import Pipeline, PipelineTypes
from models.pipeline.mixin.mixin_device import DeviceMixin
from models.pipeline.mixin.mixin_inference import InferenceMixin
from models.pipeline.block.block_model import BuildModelBlock
from models.pipeline.block.block_inference import BuildInferenceBlock
from models.pipeline.block.block_serialize import SerializeDataBlock
from models.pipeline.block.block_serialize import SerializeModelBlock


TInferPipeline = TypeVar("TInferPipeline", bound="DLInferencePipeline", covariant=True)


@Pipeline.register(PipelineTypes.DL_INFERENCE)
class DLInferencePipeline(Pipeline, DeviceMixin, InferenceMixin):
    is_built = False

    focuses = [
        BuildModelBlock,
        BuildInferenceBlock,
        SerializeDataBlock,
        SerializeModelBlock,
    ]

    def after_load(self) -> None:
        self.is_built = True
        self.data = self.serialize_data.data
        if self.serialize_model is not None:
            self.serialize_model.verbose = False