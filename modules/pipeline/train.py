import os

from abc import ABCMeta
from typing import List


class TrainingPipeline(
    Pipeline,
    _DeviceMixin,
    _EvaluationMixin,
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
        # build pipeline
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