import torch.nn as nn

from torch import Tensor
from abc import ABCMeta
from typing import Any, Optional, Tuple, TypeVar, Dict, Type

from tools.bases.register import WithRegister
from tools.utils.type import tensor_dict_type, losses_type

from models.model.constant import LABEL_KEY, LOSS_KEY, PREDICTIONS_KEY
from models.model.train_state import TrainerState


TLoss = TypeVar("TLoss", bound="ILoss", covariant=True)
loss_dict: Dict[str, Type["ILoss"]] = {}


class ILoss(nn.Module, WithRegister[TLoss], metaclass=ABCMeta):
    d = loss_dict
    placeholder_key = "[PLACEHOLDER]"

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def _reduce(self, losses: Tensor) -> Tensor:
        if self.reduction == "none":
            return losses
        if self.reduction == "mean":
            return losses.mean()
        if self.reduction == "sum":
            return losses.sum()
        raise NotImplementedError(f"reduction '{self.reduction}' is not implemented")

    # optional callbacks

    def get_forward_args(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
    ) -> Tuple[Any, ...]:
        return forward_results[PREDICTIONS_KEY], batch[LABEL_KEY]

    def postprocess(
        self,
        losses: losses_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
    ) -> tensor_dict_type:
        if not isinstance(losses, dict):
            losses = {LOSS_KEY: losses}
        return {k: self._reduce(v) for k, v in losses.items()}

    # api

    def run(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
    ) -> tensor_dict_type:
        args = self.get_forward_args(forward_results, batch, state)
        losses = self(*args)
        losses = self.postprocess(losses, batch, state)
        return losses