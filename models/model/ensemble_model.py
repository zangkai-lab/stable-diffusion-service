import torch
import torch.nn as nn

from torch import Tensor
from typing import Any, Dict, List, Optional, Protocol

from tools.utils.type import tensor_dict_type, forward_results_type
from tools.utils.safe import safe_execute
from tools.utils.icopy import get_clones

from models.model.constant import PREDICTIONS_KEY
from models.model.model_dl import IDLModel


class EnsembleFn(Protocol):
    def __call__(self, key: str, tensors: List[Tensor]) -> Tensor:
        pass


class DLEnsembleModel(nn.Module):
    ensemble_fn: Optional[EnsembleFn]

    def __init__(self, m: IDLModel, num_repeat: int) -> None:
        super().__init__()
        self.ms = get_clones(m, num_repeat)
        self.ensemble_fn = None

    def forward(self, *args: Any) -> forward_results_type:
        outputs: Dict[str, List[Tensor]] = {}
        for m in self.ms:
            m_outputs = m(*args)
            if isinstance(m_outputs, Tensor):
                m_outputs = {PREDICTIONS_KEY: m_outputs}
            for k, v in m_outputs.items():
                outputs.setdefault(k, []).append(v)
        final_results: tensor_dict_type = {}
        for k in sorted(outputs):
            if self.ensemble_fn is None:
                v = torch.stack(outputs[k]).mean(0)
            else:
                v = safe_execute(self.ensemble_fn, dict(key=k, tensors=outputs[k]))
            final_results[k] = v
        return final_results

    def run(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional["TrainerState"] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        m: IDLModel = self.ms[0]
        args = m.get_forward_args(batch_idx, batch, state, **kwargs)
        forward_results = self(*args)
        outputs = m.postprocess(batch_idx, batch, forward_results, state, **kwargs)
        return outputs
