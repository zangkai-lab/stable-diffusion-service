import torch.nn as nn
import copy

from typing import List, Union


def shallow_copy_dict(d: dict) -> dict:
    d = d.copy()
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = shallow_copy_dict(v)
    return d


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