import os

from abc import ABCMeta, abstractmethod
from typing import List, Optional, Union

from models.pipeline.pipeline_base import Pipeline, Block
from models.data.data import IData
from models.pipeline.mixin.mixin_device import DeviceMixin
from models.pipeline.mixin.mixin_evaluation import EvaluationMixin

from models.pipeline.block.block_trainer import BuildTrainerBlock
from models.pipeline.block.block_model import BuildModelBlock
from models.pipeline.block.block_inference import BuildInferenceBlock
from models.pipeline.block.block_serialize import SerializeDataBlock, SerializeModelBlock
from models.pipeline.block.block_loss import BuildLossBlock
from models.pipeline.block.block_metrics import BuildMetricsBlock
from models.pipeline.block.block_workspace import PrepareWorkplaceBlock, prepare_workspace_from
from models.pipeline.block.block_stateinfo import ExtractStateInfoBlock
from models.pipeline.block.block_monitor import BuildMonitorsBlock
from models.pipeline.block.block_callback import BuildCallbacksBlock
from models.pipeline.block.block_optimizer import BuildOptimizersBlock, SerializeOptimizerBlock
from models.pipeline.block.block_recordsampler import RecordNumSamplesBlock
from models.pipeline.block.block_report import ReportBlock
from models.pipeline.block.block_training import TrainingBlock
from models.pipeline.pipeline_base import PipelineTypes

from tools.utils.type import sample_weights_type
from tools.utils.ddp import is_local_rank_0
from tools.bases.serializable import Serializer


class TrainingPipeline(
    Pipeline,
    DeviceMixin,
    EvaluationMixin,
    metaclass=ABCMeta,
):
    is_built = False

    @property
    @abstractmethod
    def set_defaults_block(self) -> Block:
        pass

    @property
    @abstractmethod
    def set_trainer_defaults_block(self) -> Block:
        pass

    @property
    def build_trainer(self) -> BuildTrainerBlock:
        return self.get_block(BuildTrainerBlock)

    @property
    def building_blocks(self) -> List[Block]:
        return [
            self.set_defaults_block,
            PrepareWorkplaceBlock(),
            ExtractStateInfoBlock(),
            BuildLossBlock(),
            BuildModelBlock(),
            BuildMetricsBlock(),
            BuildInferenceBlock(),
            self.set_trainer_defaults_block,
            BuildMonitorsBlock(),
            BuildCallbacksBlock(),
            BuildOptimizersBlock(),
            BuildTrainerBlock(),
            RecordNumSamplesBlock(),
            ReportBlock(),
            TrainingBlock(),
            SerializeDataBlock(),
            SerializeModelBlock(),
            SerializeOptimizerBlock(),
        ]

    def after_load(self) -> None:
        self.is_built = True
        workspace = prepare_workspace_from("_logs")
        self.config.workspace = workspace

    def prepare(self, data: IData, sample_weights: sample_weights_type = None) -> None:
        self.data = data.set_sample_weights(sample_weights)
        self.training_workspace = self.config.workspace
        if not self.is_built:
            self.build(*self.building_blocks)
            self.is_built = True
        else:
            for block in self.blocks:
                block.training_workspace = self.training_workspace

    def fit(
        self,
        data: IData,
        *,
        sample_weights: sample_weights_type = None,
        cuda: Optional[Union[int, str]] = None,
    ) -> "TrainingPipeline":
        # block pipeline
        self.prepare(data, sample_weights)
        # check rank 0
        workspace = self.config.workspace if is_local_rank_0() else None
        # save data info
        if workspace is not None:
            Serializer.save(
                os.path.join(workspace, SerializeDataBlock.package_folder),
                data,
                save_npd=False,
            )
        # run pipeline
        self.run(data, cuda=cuda)
        # save pipeline
        if workspace is not None:
            pipeline_folder = DLPipelineSerializer.pipeline_folder
            DLPipelineSerializer.save(self, os.path.join(workspace, pipeline_folder))
        # return
        return self


@Pipeline.register(PipelineTypes.DL_TRAINING)
class DLTrainingPipeline(TrainingPipeline):
    @property
    def set_defaults_block(self) -> Block:
        return SetDefaultsBlock()

    @property
    def set_trainer_defaults_block(self) -> Block:
        return SetTrainerDefaultsBlock()