import torch
import numpy as np

from typing import Union
from typing import Dict
from typing import Any

arr_type = Union[np.ndarray, torch.Tensor]  # 可以是numpy数组或者torch张量
tensor_dict_type = Dict[str, Union[torch.Tensor, Any]]  # 可以是torch张量或者任意类型的字典
