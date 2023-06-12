import torch.nn as nn

from typing import Optional, Any, List

from tools.utils.safe import check_requires, filter_kw, get_num_positional_args
from tools.utils.type import tensor_dict_type

from models.model.constant import PREDICTIONS_KEY


def _forward(
    m: nn.Module,
    batch_idx: int,
    batch: tensor_dict_type,
    general_input_key: str,
    state: Optional["TrainerState"] = None,
    *,
    general_output_key: str = PREDICTIONS_KEY,
    **kwargs: Any,
) -> tensor_dict_type:
    fn = m.forward
    if check_requires(fn, "general_output_key"):
        kwargs["general_output_key"] = general_output_key
    kw = filter_kw(fn, kwargs)
    args: List[Any] = []
    if check_requires(fn, "batch_idx"):
        args.append(batch_idx)
    if get_num_positional_args(fn) > 0:
        args.append(batch if check_requires(fn, "batch") else batch[general_input_key])
    if check_requires(fn, "state"):
        args.append(state)
    rs = m(*args, **kw)
    if not isinstance(rs, dict):
        rs = {general_output_key: rs}
    return rs