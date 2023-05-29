# inspect是Python的内建模块，它提供了很多有用的函数来帮助获取对象的信息，比如模块、类、函数、追踪记录、帧以及协程。
# 这些信息通常包括类的成员、文档字符串、源代码、规格以及函数的参数等
import os
import inspect
import json
import torch
import numpy as np
import copy
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Callable, Dict, Protocol, TypeVar, Optional, NamedTuple, Union, List

SCORES_FILE = "scores.json"

tensor_dict_type = Dict[str, Union[torch.Tensor, Any]]
np_dict_type = Dict[str, Union[np.ndarray, Any]]
arr_type = Union[np.ndarray, torch.Tensor]


def shallow_copy_dict(d: dict) -> dict:
    d = d.copy()
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = shallow_copy_dict(v)
    return d


TFnResponse = TypeVar("TFnResponse")


def check_requires(fn: Any, name: str, strict: bool = True) -> bool:
    if isinstance(fn, type):
        fn = fn.__init__  # type: ignore
    signature = inspect.signature(fn)
    for k, param in signature.parameters.items():
        if not strict and param.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if k == name:
            if param.kind is inspect.Parameter.VAR_POSITIONAL:
                return False
            return True
    return False


def filter_kw(
    fn: Callable,
    kwargs: Dict[str, Any],
    *,
    strict: bool = False,
) -> Dict[str, Any]:
    kw = {}
    for k, v in kwargs.items():
        if check_requires(fn, k, strict):
            kw[k] = v
    return kw


class Fn(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> TFnResponse:
        pass


# 用于执行函数，过滤掉不需要的参数
def safe_execute(fn: Fn, kw: Dict[str, Any], *, strict: bool = False) -> TFnResponse:
    return fn(**filter_kw(fn, kw, strict=strict))


def update_dict(src_dict: dict, tgt_dict: dict) -> dict:
    for k, v in src_dict.items():
        tgt_v = tgt_dict.get(k)
        if tgt_v is None:
            tgt_dict[k] = v
        elif not isinstance(v, dict):
            tgt_dict[k] = v
        else:
            update_dict(v, tgt_v)
    return tgt_dict


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def is_int(arr: np.ndarray) -> bool:
    return np.issubdtype(arr.dtype, np.integer)


def is_float(arr: np.ndarray) -> bool:
    return np.issubdtype(arr.dtype, np.floating)


def to_standard(arr: np.ndarray) -> np.ndarray:
    if is_int(arr):
        arr = arr.astype(np.int64)
    elif is_float(arr):
        arr = arr.astype(np.float32)
    return arr


def to_torch(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(to_standard(arr))


class DDPInfo(NamedTuple):
    rank: int
    world_size: int
    local_rank: int


# 获取分布式数据并行的信息
def get_ddp_info() -> Optional[DDPInfo]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        return DDPInfo(rank, world_size, local_rank)
    return None


def is_local_rank_0() -> bool:
    ddp_info = get_ddp_info()
    return ddp_info is None or ddp_info.local_rank == 0


def fix_denormal_states(
    states: tensor_dict_type,
    *,
    eps: float = 1.0e-32,
    verbose: bool = False,
) -> tensor_dict_type:
    new_states = shallow_copy_dict(states)
    num_total = num_denormal_total = 0
    for k, v in states.items():
        if not v.is_floating_point():
            continue
        num_total += v.numel()
        denormal = (v != 0) & (v.abs() < eps)
        num_denormal = denormal.sum().item()
        num_denormal_total += num_denormal
        if num_denormal > 0:
            new_states[k][denormal] = v.new_zeros(num_denormal)
    if verbose:
        print(f"denormal ratio : {num_denormal_total / num_total:8.6f}")
    return new_states


def get_clones(
    module: nn.Module,
    n: int,
    *,
    return_list: bool = False,
) -> Union[nn.ModuleList, List[nn.Module]]:
    module_list = [module]
    for _ in range(n - 1):
        module_list.append(copy.deepcopy(module))
    if return_list:
        return module_list
    return nn.ModuleList(module_list)


def get_world_size() -> int:
    ddp_info = get_ddp_info()
    return 1 if ddp_info is None else ddp_info.world_size


def sigmoid(arr: arr_type) -> arr_type:
    if isinstance(arr, np.ndarray):
        return 1.0 / (1.0 + np.exp(-arr))
    return torch.sigmoid(arr)


def softmax(arr: arr_type) -> arr_type:
    if isinstance(arr, np.ndarray):
        logits = arr - np.max(arr, axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(1, keepdims=True)
    return F.softmax(arr, dim=1)


def to_device(
    batch: tensor_dict_type,
    device: torch.device,
    **kwargs: Any,
) -> tensor_dict_type:
    return {
        k: v.to(device, **kwargs)
        if isinstance(v, torch.Tensor)
        else [
            vv.to(device, **kwargs) if isinstance(vv, torch.Tensor) else vv for vv in v
        ]
        if isinstance(v, list)
        else v
        for k, v in batch.items()
    }


def is_string(arr: np.ndarray) -> bool:
    return np.issubdtype(arr.dtype, str)


def np_batch_to_tensor(np_batch: np_dict_type) -> tensor_dict_type:
    return {
        k: v if not isinstance(v, np.ndarray) or is_string(v) else to_torch(v)
        for k, v in np_batch.items()
    }