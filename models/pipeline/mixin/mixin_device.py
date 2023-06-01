import torch


# Device类，用于获取模型的设备信息
class _DeviceMixin:
    build_model: BuildModelBlock

    @property
    def device(self) -> torch.device:
        return self.build_model.model.device
