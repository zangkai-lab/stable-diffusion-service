import numpy as np
import torch

from tools.utils.type import arr_type


def is_int(arr: np.ndarray) -> bool:
    return np.issubdtype(arr.dtype, np.integer)


def is_float(arr: np.ndarray) -> bool:
    return np.issubdtype(arr.dtype, np.floating)


def is_string(arr: np.ndarray) -> bool:
    return np.issubdtype(arr.dtype, str)


def sigmoid(arr: arr_type) -> arr_type:
    if isinstance(arr, np.ndarray):
        return 1.0 / (1.0 + np.exp(-arr))
    return torch.sigmoid(arr)