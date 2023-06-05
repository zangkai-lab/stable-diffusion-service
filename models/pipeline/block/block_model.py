from typing import Any, Dict, Union

from models.pipeline.pipeline_base import Block

from tools.utils.safe import safe_execute
from tools.mixin.inject_default import InjectDefaultsMixin

from models.model.model_dl import IDLModel
from models.config.train_config import DLConfig, MLConfig
from models.model.ensemble_model import DLEnsembleModel


@Block.register("build_model")
class BuildModelBlock(InjectDefaultsMixin, Block):
    model: IDLModel

    def build(self, config: Union[DLConfig, MLConfig]) -> None:
        model_name = config.model_name
        model_config = config.model_config or {}
        if isinstance(config, MLConfig):
            self._setup_ml_model(config, model_config)
        num_repeat = config.num_repeat
        m = safe_execute(IDLModel.get(model_name), model_config)
        if num_repeat is None:
            self.model = m
        else:
            self.model = DLEnsembleModel(m, num_repeat)

    def _setup_ml_model(self, config: MLConfig, model_config: Dict[str, Any]) -> None:
        if config.encoder_settings is None or config.index_mapping is None:
            encoder_settings = config.encoder_settings
        else:
            encoder_settings = {}
            for k, v in config.encoder_settings.items():
                encoder_settings[str(config.index_mapping[k])] = v
        model_config["encoder_settings"] = encoder_settings
        model_config["global_encoder_settings"] = config.global_encoder_settings
        mc = self._defaults.setdefault("model_config", {})
        if encoder_settings is not None:
            d = {k: v.asdict() for k, v in encoder_settings.items()}
            mc["encoder_settings"] = d
        if config.global_encoder_settings is not None:
            ges = config.global_encoder_settings.asdict()
            self._defaults["global_encoder_settings"] = ges
        self._defaults["index_mapping"] = config.index_mapping