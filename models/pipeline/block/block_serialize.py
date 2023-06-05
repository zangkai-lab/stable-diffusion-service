import os
import json
import torch
import shutil

from typing import Optional, Dict, List, Type

from models.pipeline.pipeline_base import Block
from models.config.train_config import DLConfig
from models.data.data import IData
from models.pipeline.block.block_model import BuildModelBlock
from models.model.constant import CHECKPOINTS_FOLDER, SCORES_FILE, PT_PREFIX
from models.utils.scores import get_scores, get_sorted_checkpoints

from tools.bases.serializable import Serializer


@Block.register("serialize_data")
class SerializeDataBlock(Block):
    data: Optional[IData]
    config: DLConfig
    package_folder: str = "data_module"

    def build(self, config: DLConfig) -> None:
        self.data = None
        self.config = config

    def save_extra(self, folder: str) -> None:
        if not self.is_local_rank_0:
            return
        if self.training_workspace is not None:
            data_folder = os.path.join(self.training_workspace, self.package_folder)
            shutil.copytree(data_folder, folder)
        elif self.data is not None:
            Serializer.save(folder, self.data, save_npd=False)

    def load_from(self, folder: str) -> None:
        if os.path.isdir(folder):
            self.data = Serializer.load(folder, IData, load_npd=False)


@Block.register("serialize_model")
class SerializeModelBlock(Block):
    config: DLConfig

    verbose: bool = True
    ckpt_folder: Optional[str] = None
    ckpt_scores: Optional[Dict[str, float]] = None

    def build(self, config: DLConfig) -> None:
        self.config = config
        self.best_score = 0.0

    @property
    def requirements(self) -> List[Type[Block]]:
        return [BuildModelBlock]

    @property
    def build_model(self) -> BuildModelBlock:
        return self.get_previous(BuildModelBlock)

    def save_extra(self, folder: str) -> None:
        if not self.is_local_rank_0:
            return
        warn_msg = "no checkpoints found at {}, current model states will be saved"
        if self.training_workspace is not None:
            ckpt_folder = os.path.join(self.training_workspace, CHECKPOINTS_FOLDER)
            if get_sorted_checkpoints(ckpt_folder):
                shutil.copytree(ckpt_folder, folder)
            else:
                if self.verbose:
                    print(warn_msg.format(ckpt_folder))
                self._save_current(folder)
            return
        if self.ckpt_folder is None or self.ckpt_scores is None:
            if self.verbose:
                print("current model states will be saved")
            self._save_current(folder)
        else:
            any_saved = False
            filtered_scores = {}
            os.makedirs(folder, exist_ok=True)
            for file, score in self.ckpt_scores.items():
                ckpt_path = os.path.join(self.ckpt_folder, file)
                if not os.path.isfile(ckpt_path):
                    if self.verbose:
                        msg = f"cannot find checkpoint at '{ckpt_path}', did you delete it?"
                        print(msg)
                    continue
                any_saved = True
                filtered_scores[file] = score
                shutil.copyfile(ckpt_path, os.path.join(folder, file))
            if any_saved:
                with open(os.path.join(folder, SCORES_FILE), "w") as f:
                    json.dump(filtered_scores, f)
            else:
                if self.verbose:
                    print(warn_msg.format(self.ckpt_folder))
                self._save_current(folder)

    def load_from(self, folder: str) -> None:
        model = self.build_model.model
        best_file = get_sorted_checkpoints(folder)[0]
        model.load_state_dict(torch.load(os.path.join(folder, best_file)))
        scores = get_scores(folder)
        self.ckpt_folder = folder
        self.ckpt_scores = scores

    def _save_current(self, folder: str) -> None:
        os.makedirs(folder, exist_ok=True)
        latest_file = f"{PT_PREFIX}-1.pt"
        latest_path = os.path.join(folder, latest_file)
        new_scores_path = os.path.join(folder, SCORES_FILE)
        torch.save(self.build_model.model.state_dict(), latest_path)
        with open(new_scores_path, "w") as f:
            json.dump({latest_file: 0.0}, f)