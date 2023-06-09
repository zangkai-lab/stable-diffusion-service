import torch
import torch.nn as nn

from torch.optim import Optimizer
from typing import Any, ContextManager, Optional, Dict


class mode_context:
    """
    Help entering specific mode and recovering previous mode

    This is a context controller for entering specific mode at the beginning
    and back to previous mode at the end.

    Parameters
    ----------
    module : nn.Module, arbitrary PyTorch module.

    Examples
    --------
    >>> module = nn.Module()
    >>> with mode_context(module):
    >>>     pass  # do something

    """

    def __init__(
        self,
        module: nn.Module,
        *,
        to_train: Optional[bool],
        use_grad: Optional[bool],
        use_inference: Optional[bool] = None,
    ):
        self._to_train = to_train
        self._module, self._training = module, module.training
        self._cache = {p: p.requires_grad for p in module.parameters()}
        if use_grad is not None:
            for p in module.parameters():
                p.requires_grad_(use_grad)
        if use_grad is None:
            self._grad_context: Optional[ContextManager] = None
        else:
            self._grad_context = torch.enable_grad() if use_grad else torch.no_grad()
        if use_inference is None:
            self._inference_context: Optional[ContextManager] = None
        else:
            self._inference_context = torch.inference_mode(use_inference)

    def __enter__(self) -> None:
        if self._to_train is not None:
            self._module.train(mode=self._to_train)
        if self._grad_context is not None:
            self._grad_context.__enter__()
        if self._inference_context is not None:
            self._inference_context.__enter__()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._to_train is not None:
            self._module.train(mode=self._training)
        if self._inference_context is not None:
            self._inference_context.__exit__(exc_type, exc_val, exc_tb)
        if self._grad_context is not None:
            self._grad_context.__exit__(exc_type, exc_val, exc_tb)
        for p, v in self._cache.items():
            if p.requires_grad != v:
                p.requires_grad_(v)


class train_context(mode_context):
    """
    Useful when we need to get gradients with our PyTorch model during evaluating.
    """

    def __init__(self, module: nn.Module, *, use_grad: bool = True):
        super().__init__(module, to_train=True, use_grad=use_grad, use_inference=False)


class eval_context(mode_context):
    """
    Useful when we need to predict something with our PyTorch model during training.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        use_grad: Optional[bool] = False,
        use_inference: Optional[bool] = None,
    ):
        if use_inference is None and use_grad is not None:
            use_inference = not use_grad
        super().__init__(
            module,
            to_train=False,
            use_grad=use_grad,
            use_inference=use_inference,
        )


class no_grad_context(torch.no_grad):
    def __init__(self, *, enabled: bool):
        super().__init__()
        self.enabled = enabled

    def __enter__(self) -> None:
        if not self.enabled:
            return
        super().__enter__()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if not self.enabled:
            return
        super().__exit__(exc_type, exc_val, exc_tb)


class toggle_optimizer:
    """
    Help focusing gradients on specific optimizer and recovering previous states

    This is a context controller for requiring and only requiring grads for parameters
    of the given optimizer at the beginning, and back to previous grads requiring states
    at the end.

    Examples
    --------
    >>> module = nn.Module()
    >>> optimizer = torch.optim.Adam()
    >>> with toggle_optimizer(module, optimizer):
    >>>     pass  # do something

    """

    def __init__(self, m: nn.Module, optimizer: Optimizer, *, enabled: bool = True):
        self.m = m
        self.optimizer = optimizer
        self.enabled = enabled
        self.requires_grad: Dict[str, bool] = {}

    def __enter__(self) -> None:
        if not self.enabled:
            return
        self.requires_grad = {k: p.requires_grad for k, p in self.m.named_parameters()}
        for p in self.m.parameters():
            p.requires_grad = False
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                p.requires_grad = True

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if not self.enabled:
            return
        for k, p in self.m.named_parameters():
            requires_grad = self.requires_grad.get(k)
            if requires_grad is not None:
                p.requires_grad = requires_grad