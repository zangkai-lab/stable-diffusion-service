import torch
import torch.nn as nn

from torch.cuda.amp.autocast_mode import autocast
from typing import Any, Type, TypeVar, Optional, Union

from .utils import is_cpu
from .utils import empty_cuda_cache
from .utils import get_device

from modules.pipeline import TrainingPipeline, DLInferencePipeline


TPipeline = Union[TrainingPipeline, DLInferencePipeline]
TAPI = TypeVar("TAPI", bound="APIMixin")


class APIMixin:
    m: nn.Module
    device: torch.device
    use_amp: bool  # 自动混合精度
    use_half: bool  # 半精度

    def __init__(
        self,
        m: nn.Module,
        device: torch.device,
        *,
        use_amp: bool = False,
        use_half: bool = False,
    ):
        if use_amp and use_half:
            raise ValueError("`use_amp` & `use_half` should not be True simultaneously")
        self.m = m.eval().requires_grad_(False)  # 进入部署模式并关闭梯度计算
        self.device = device
        self.use_amp = use_amp
        self.use_half = use_half

    # 模型部署：拓展了原本的to方法，增加了use_amp和use_half两个参数
    def to(
        self,
        device: torch.device,
        *,
        use_amp: bool = False,
        use_half: bool = False,
    ) -> None:
        if use_amp and use_half:
            raise ValueError("`use_amp` & `use_half` should not be True simultaneously")
        self.device = device
        self.use_amp = use_amp
        self.use_half = use_half
        device_is_cpu = is_cpu(device)
        if device_is_cpu:
            self.m.to(device)
        if use_half:
            self.m.half()
        else:
            self.m.float()
        if not device_is_cpu:
            self.m.to(device)

    def empty_cuda_cache(self) -> None:
        empty_cuda_cache(self.device)

    # models.to(device)和model.half()这样的方法用于改变模型的存储设备和数据类型。
    # 然而，这并不意味着在执行计算时，所有操作都会按照模型的数据类型进行。
    @property
    def amp_context(self) -> autocast:
        return autocast(enabled=self.use_amp)

    # 用来从管道对象创建 APIMixin 对象
    # 而 to 是用来修改已经存在的 APIMixin 对象的
    @classmethod
    def from_pipeline(
        cls: Type[TAPI],
        m: TPipeline,
        device: Optional[str] = None,
        *,
        use_amp: bool = False,
        use_half: bool = False,
        **kwargs: Any,
    ) -> TAPI:
        if use_amp and use_half:
            raise ValueError("`use_amp` & `use_half` should not be True simultaneously")
        model = m.build_model.model
        if use_half:
            model.half()
        if device is not None:
            model.to(device)
        return cls(
            model,
            get_device(model),
            use_amp=use_amp,
            use_half=use_half,
            **kwargs,
        )
