import numpy as np
import os
import shutil
import torch

from tqdm import tqdm
from tempfile import mkdtemp
from collections import OrderedDict
from tempfile import TemporaryDirectory
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union

from tools.utils.type import tensor_dict_type, states_callback_type
from tools.bases.serializable import Serializer

from models.config.train_config import DLConfig
from models.data.data_loader import IDataLoader
from models.pipeline.pipeline_base import Block, Pipeline
from models.utils.workspace import get_workspace
from models.utils.scores import get_sorted_checkpoints, get_scores
from models.pipeline.block.block_workspace import PrepareWorkplaceBlock
from models.pipeline.block.block_serialize import SerializeModelBlock
from models.model.saving import Saving


class PackType(str, Enum):
    TRAINING = "training"
    INFERENCE = "inference"
    EVALUATION = "evaluation"


class DLPipelineSerializer:
    id_file = "id.txt"
    config_file = "config.json"
    blocks_file = "blocks.json"
    pipeline_folder = "pipeline"

    # api

    @classmethod
    def save(cls, pipeline: Pipeline, folder: str, *, compress: bool = False) -> None:
        original_folder = None
        if compress:
            original_folder = folder
            folder = mkdtemp()
        Serializer.save(folder, pipeline)
        for block in pipeline.blocks:
            block.save_extra(os.path.join(folder, block.__identifier__))
        if compress and original_folder is not None:
            abs_folder = os.path.abspath(folder)
            abs_original = os.path.abspath(original_folder)
            Saving.compress(abs_folder)
            shutil.move(f"{abs_folder}.zip", f"{abs_original}.zip")

    @classmethod
    def pack(
        cls,
        workspace: str,
        export_folder: str,
        *,
        pack_type: PackType = PackType.INFERENCE,
        compress: bool = True,
    ) -> None:
        if pack_type == PackType.TRAINING:
            swap_id = None
            focuses = None
            excludes = [PrepareWorkplaceBlock]
        elif pack_type == PackType.INFERENCE:
            swap_id = DLInferencePipeline.__identifier__
            focuses = DLInferencePipeline.focuses
            excludes = None
        elif pack_type == PackType.EVALUATION:
            swap_id = DLEvaluationPipeline.__identifier__
            focuses = DLEvaluationPipeline.focuses
            excludes = None
        else:
            raise ValueError(f"unrecognized `pack_type` '{pack_type}' occurred")
        pipeline_folder = os.path.join(workspace, cls.pipeline_folder)
        pipeline = cls._load(
            pipeline_folder,
            swap_id=swap_id,
            focuses=focuses,
            excludes=excludes,
        )
        cls.save(pipeline, export_folder, compress=compress)

    @classmethod
    def pack_and_load_inference(cls, workplace: str) -> DLInferencePipeline:
        with TemporaryDirectory() as tmp_folder:
            cls.pack(
                workplace,
                export_folder=tmp_folder,
                pack_type=PackType.INFERENCE,
                compress=False,
            )
            return cls.load_inference(tmp_folder)

    @classmethod
    def pack_onnx(
        cls,
        workplace: str,
        export_file: str = "model.onnx",
        dynamic_axes: Optional[Union[List[int], Dict[int, str]]] = None,
        *,
        input_sample: Optional[tensor_dict_type] = None,
        loader_sample: Optional[IDataLoader] = None,
        opset: int = 11,
        simplify: bool = True,
        num_samples: Optional[int] = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> DLInferencePipeline:
        if input_sample is None and loader_sample is None:
            msg = "either `input_sample` or `loader_sample` should be provided"
            raise ValueError(msg)
        m = cls.pack_and_load_inference(workplace)
        model = m.build_model.model
        if input_sample is None:
            input_sample = get_input_sample(loader_sample, get_device(model))  # type: ignore
        model.to_onnx(
            export_file,
            input_sample,
            dynamic_axes,
            opset=opset,
            simplify=simplify,
            num_samples=num_samples,
            verbose=verbose,
            **kwargs,
        )
        return m

    @classmethod
    def pack_scripted(
        cls,
        workplace: str,
        export_file: str = "model.pt",
    ) -> DLInferencePipeline:
        m = cls.pack_and_load_inference(workplace)
        model = torch.jit.script(m.build_model.model)
        torch.jit.save(model, export_file)
        return m

    @classmethod
    def fuse_inference(
        cls,
        src_folders: List[str],
        *,
        cuda: Optional[str] = None,
        num_picked: Optional[Union[int, float]] = None,
        states_callback: states_callback_type = None,
    ) -> DLInferencePipeline:
        return cls._fuse_multiple(
            src_folders,
            PackType.INFERENCE,
            cuda,
            num_picked,
            states_callback,
        )

    @classmethod
    def fuse_evaluation(
        cls,
        src_folders: List[str],
        *,
        cuda: Optional[str] = None,
        num_picked: Optional[Union[int, float]] = None,
        states_callback: states_callback_type = None,
    ) -> DLEvaluationPipeline:
        return cls._fuse_multiple(
            src_folders,
            PackType.EVALUATION,
            cuda,
            num_picked,
            states_callback,
        )

    @classmethod
    def load_training(cls, folder: str) -> TrainingPipeline:
        return cls._load(folder, swap_id=DLTrainingPipeline.__identifier__)

    @classmethod
    def load_inference(cls, folder: str) -> DLInferencePipeline:
        return cls._load_inference(folder)

    @classmethod
    def load_evaluation(cls, folder: str) -> DLEvaluationPipeline:
        return cls._load_evaluation(folder)

    # internal

    @classmethod
    def _load(
        cls,
        folder: str,
        *,
        swap_id: Optional[str] = None,
        focuses: Optional[List[Type[Block]]] = None,
        excludes: Optional[List[Type[Block]]] = None,
    ) -> Pipeline:
        with get_workspace(folder) as workspace:
            # handle info
            info = Serializer.load_info(workspace)
            if focuses is not None or excludes is not None:
                if focuses is None:
                    focuses_set = None
                else:
                    focuses_set = {b.__identifier__ for b in focuses}
                block_types = info["blocks"]
                if focuses_set is not None:
                    block_types = [b for b in block_types if b in focuses_set]
                    left = sorted(focuses_set - set(block_types))
                    if left:
                        raise ValueError(
                            "following blocks are specified in `focuses` "
                            f"but not found in the loaded blocks: {', '.join(left)}"
                        )
                if excludes is not None:
                    excludes_set = {b.__identifier__ for b in excludes}
                    block_types = [b for b in block_types if b not in excludes_set]
                info["blocks"] = block_types
            # load
            pipeline = Serializer.load_empty(workspace, Pipeline, swap_id=swap_id)
            pipeline.serialize_folder = workspace
            if info is None:
                info = Serializer.load_info(workspace)
            pipeline.from_info(info)
            for block in pipeline.blocks:
                block.load_from(os.path.join(workspace, block.__identifier__))
            pipeline.after_load()
        return pipeline

    @classmethod
    def _load_inference(
        cls,
        folder: str,
        excludes: Optional[List[Type[Block]]] = None,
    ) -> DLInferencePipeline:
        return cls._load(
            folder,
            swap_id=DLInferencePipeline.__identifier__,
            focuses=DLInferencePipeline.focuses,
            excludes=excludes,
        )

    @classmethod
    def _load_evaluation(
        cls,
        folder: str,
        excludes: Optional[List[Type[Block]]] = None,
    ) -> DLEvaluationPipeline:
        return cls._load(
            folder,
            swap_id=DLEvaluationPipeline.__identifier__,
            focuses=DLEvaluationPipeline.focuses,
            excludes=excludes,
        )

    @classmethod
    def _fuse_multiple(
        cls,
        src_folders: List[str],
        pack_type: PackType,
        cuda: Optional[str] = None,
        num_picked: Optional[Union[int, float]] = None,
        states_callback: states_callback_type = None,
    ) -> DLInferencePipeline:
        if pack_type == PackType.TRAINING:
            raise ValueError("should not pack to training pipeline when fusing")
        # get num picked
        num_total = num_repeat = len(src_folders)
        if num_picked is not None:
            if isinstance(num_picked, float):
                if num_picked < 0.0 or num_picked > 1.0:
                    raise ValueError("`num_picked` should ∈ [0, 1] when set to float")
                num_picked = round(num_total * num_picked)
            if num_picked < 1:
                raise ValueError("calculated `num_picked` should be at least 1")
            scores = []
            for i, folder in enumerate(src_folders):
                ckpt_folder = os.path.join(folder, SerializeModelBlock.__identifier__)
                folder_scores = get_scores(ckpt_folder)
                scores.append(max(folder_scores.values()))
            scores_array = np.array(scores)
            picked_indices = np.argsort(scores)[::-1][:num_picked]
            src_folders = [src_folders[i] for i in picked_indices]
            original_score = scores_array.mean().item()
            picked_score = scores_array[picked_indices].mean().item()
            print(
                f"picked {num_picked} / {num_total}, "
                f"score: {original_score} -> {picked_score}"
            )
            num_repeat = num_picked
        # get empty pipeline
        with get_workspace(src_folders[0], force_new=True) as workspace:
            info = Serializer.load_info(workspace)
            config: DLConfig = DLConfig.from_pack(info["config"])
            config.num_repeat = num_repeat
            info["config"] = config.to_pack().asdict()
            Serializer.save_info(workspace, info=info)
            fn = (
                cls._load_inference
                if pack_type == PackType.INFERENCE
                else cls._load_evaluation
            )
            # avoid loading model because the ensembled model has different states
            m = fn(workspace, excludes=[SerializeModelBlock])
            # but we need to build the SerializeModelBlock again for further save/load
            b_serialize_model = SerializeModelBlock()
            b_serialize_model.verbose = False
            m.build(b_serialize_model)
        # merge state dict
        merged_states: OrderedDict[str, torch.Tensor] = OrderedDict()
        for i, folder in enumerate(tqdm(src_folders, desc="fuse")):
            with get_workspace(folder) as i_folder:
                ckpt_folder = os.path.join(i_folder, SerializeModelBlock.__identifier__)
                checkpoints = get_sorted_checkpoints(ckpt_folder)
                checkpoint_path = os.path.join(ckpt_folder, checkpoints[0])
                states = torch.load(checkpoint_path, map_location=cuda)
            current_keys = list(states.keys())
            for k, v in list(states.items()):
                states[f"ms.{i}.{k}"] = v
            for k in current_keys:
                states.pop(k)
            if states_callback is not None:
                states = states_callback(m, states)
            merged_states.update(states)
        # load state dict
        model = m.build_model.model
        model.to(cuda)
        model.load_state_dict(merged_states)
        return m