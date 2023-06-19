import torch

from typing import Union
from safetensors.torch import load_file

from tools.utils.type import tensor_dict_type
from tools.utils.icopy import shallow_copy_dict


def get_tensors(inp: Union[str, tensor_dict_type]) -> tensor_dict_type:
    if isinstance(inp, str):
        if inp.endswith(".safetensors"):
            inp = load_file(inp)
        else:
            inp = torch.load(inp, map_location="cpu")
    if "state_dict" in inp:
        inp = inp["state_dict"]
    return shallow_copy_dict(inp)