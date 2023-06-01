import torch
import numpy as np

from typing import Union
from typing import Dict
from typing import Any
from typing import Optional
from typing import List

arr_type = Union[np.ndarray, torch.Tensor]  # 可以是numpy数组或者torch张量

tensor_dict_type = Dict[str, Union[torch.Tensor, Any]]  # 可以是torch张量或者任意类型的字典
np_dict_type = Dict[str, Union[np.ndarray, Any]]  # 可以是numpy数组或者任意类型的字典

"""
configs_type 可以是以下四种类型中的一种：
1. None，因为 Optional 类型表示该变量可以为 None；
2. 一个 Dict[str, Any]，表示一个键为字符串，值为任何类型的字典；
3. 一个 List[Dict[str, Any]]，表示一个元素为 Dict[str, Any] 类型的列表；
4. 不提供（即默认为 None），因为 Optional 类型允许变量不被明确赋值。
"""
configs_type = Optional[Union[List[Dict[str, Any]], Dict[str, Any]]]

