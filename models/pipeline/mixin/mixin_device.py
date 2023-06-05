import torch

from models.pipeline.block.block_model import BuildModelBlock


# Device类，用于获取模型的设备信息
class DeviceMixin:
    build_model: BuildModelBlock

    @property
    def device(self) -> torch.device:
        return self.build_model.model.device
