from typing import Any
from torch.nn import Module

from models.model.blocks.hijacks import HijackConv1d
from models.model.blocks.hijacks import HijackConv2d
from models.model.blocks.hijacks import HijackConv3d


def conv_nd(n: int, *args: Any, **kwargs: Any) -> Module:
    if n == 1:
        return HijackConv1d(*args, **kwargs)
    elif n == 2:
        return HijackConv2d(*args, **kwargs)
    elif n == 3:
        return HijackConv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {n}")