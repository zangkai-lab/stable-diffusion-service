import torch
import numpy as np

from typing import Union
from typing import Dict
from typing import Any
from typing import Optional
from typing import List
from typing import Tuple

arr_type = Union[np.ndarray, torch.Tensor]  # 可以是numpy数组或者torch张量
data_type = Optional[Union[np.ndarray, str]]  # 可以是numpy数组或者字符串
configs_type = Optional[Union[List[Dict[str, Any]], Dict[str, Any]]]  # 可以是字典或者字典列表
sample_weights_type = Optional[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]

tensor_dict_type = Dict[str, Union[torch.Tensor, Any]]  # 可以是torch张量或者任意类型的字典
np_dict_type = Dict[str, Union[np.ndarray, Any]]  # 可以是numpy数组或者任意类型的字典

forward_results_type = Union[torch.Tensor, tensor_dict_type]
losses_type = Union[torch.Tensor, tensor_dict_type]

