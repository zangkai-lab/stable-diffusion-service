import torch.nn.functional as F
import numpy as np
import torch

from tools.utils.type import arr_type


def softmax(arr: arr_type) -> arr_type:
    if isinstance(arr, np.ndarray):
        logits = arr - np.max(arr, axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(1, keepdims=True)
    return F.softmax(arr, dim=1)


def sigmoid(arr: arr_type) -> arr_type:
    if isinstance(arr, np.ndarray):
        return 1.0 / (1.0 + np.exp(-arr))
    return torch.sigmoid(arr)
