import torch

from typing import Type, List

from models.config.train_config import DLConfig
from models.utils.workspace import get_environ_workspace
from models.model.loss import loss_dict, ILoss
from models.model.custom import ModelWithCustomSteps
from models.model.model_dl import IDLModel
from models.model.trainer import callback_dict
from models.pipeline.pipeline_base import Block
from models.pipeline.block.block_model import BuildModelBlock

from tools.mixin.inject_default import InjectDefaultsMixin


@Block.register("set_defaults")
class SetDefaultsBlock(InjectDefaultsMixin, Block):
    def build(self, config: DLConfig) -> None:
        loss_name = config.loss_name
        model_name = config.model_name
        state_config = config.state_config
        callback_names = config.callback_names
        if loss_name is None:
            if model_name in loss_dict:
                loss_name = model_name
            else:
                model_base = IDLModel.get(model_name)
                if config.allow_no_loss or issubclass(model_base, ModelWithCustomSteps):
                    loss_name = ILoss.placeholder_key
                else:
                    raise ValueError(
                        "`loss_name` should be provided when "
                        f"`{model_name}` has not implemented its own loss "
                        "and `allow_no_loss` is False"
                    )
            self._defaults["loss_name"] = loss_name
        if state_config is None:
            state_config = {}
        if "max_snapshot_file" not in state_config:
            state_config["max_snapshot_file"] = 25
            self._defaults["max_snapshot_file"] = 25
        if callback_names is None:
            if model_name in callback_dict:
                callback_names = model_name
                self._defaults["callback_names"] = callback_names
        environ_workspace = get_environ_workspace()
        if environ_workspace:
            config.workspace = environ_workspace
        config.loss_name = loss_name
        config.model_name = model_name
        config.state_config = state_config
        config.callback_names = callback_names
        torch.backends.cudnn.benchmark = config.cudnn_benchmark
        # tqdm settings
        tqdm_settings = config.tqdm_settings
        if tqdm_settings is None:
            tqdm_settings = {}
        use_tqdm = tqdm_settings.setdefault("use_tqdm", False)
        tqdm_settings.setdefault("use_step_tqdm", use_tqdm)
        tqdm_settings.setdefault("use_tqdm_in_validation", False)
        tqdm_settings.setdefault("in_distributed", False)
        tqdm_settings.setdefault("tqdm_position", 0)
        tqdm_settings.setdefault("tqdm_desc", "epoch")
        config.tqdm_settings = tqdm_settings


@Block.register("set_trainer_defaults")
class SetTrainerDefaultsBlock(InjectDefaultsMixin, Block):
    def build(self, config: DLConfig) -> None:
        model = self.build_model.model
        model.permute_trainer_config(config)
        # set some trainer defaults to deep learning tasks which work well in practice
        if config.monitor_names is None:
            config.monitor_names = "conservative"
            self._defaults["monitor_names"] = "conservative"
        model_name = config.model_name
        tqdm_settings = config.tqdm_settings
        callback_names = config.callback_names
        callback_configs = config.callback_configs
        if callback_names is None:
            callback_names = []
        if callback_configs is None:
            callback_configs = {}
        if isinstance(callback_names, str):
            callback_names = [callback_names]
        auto_callback = config.auto_callback
        if "mlflow" in callback_names and auto_callback:
            mlflow_config = callback_configs.setdefault("mlflow", {})
            if "experiment_name" not in mlflow_config:
                mlflow_config["experiment_name"] = model_name
                self._defaults["mlflow_experiment_name"] = model_name
        if "_log_metrics_msg" not in callback_names and auto_callback:
            self._defaults["additional_callbacks"] = ["_log_metrics_msg"]
            callback_names.insert(0, "_log_metrics_msg")
            verbose = False
            if tqdm_settings is None or not tqdm_settings.get("use_tqdm", False):
                verbose = True
            log_metrics_msg_config = callback_configs.setdefault("_log_metrics_msg", {})
            if "verbose" not in log_metrics_msg_config:
                log_metrics_msg_config["verbose"] = verbose
                self._defaults["log_metrics_msg_verbose"] = verbose
        config.tqdm_settings = tqdm_settings
        config.callback_names = callback_names
        config.callback_configs = callback_configs

    @property
    def requirements(self) -> List[Type[Block]]:
        return [BuildModelBlock]

    @property
    def build_model(self) -> BuildModelBlock:
        return self.get_previous(BuildModelBlock)