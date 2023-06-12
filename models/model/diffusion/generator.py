import torch
import random

from torch import Tensor
from abc import ABCMeta, abstractmethod
from typing import Optional, Any

from tools.utils.safe import safe_execute
from tools.utils.icopy import shallow_copy_dict
from tools.utils.device import get_device


from models.model.constant import INPUT_KEY, LABEL_KEY, PREDICTIONS_KEY
from models.model.diffusion.utils import slerp
from models.model.forward import _forward


class GeneratorMixin(metaclass=ABCMeta):
    latent_dim: int
    num_classes: Optional[int] = None

    # inherit

    @property
    @abstractmethod
    def can_reconstruct(self) -> bool:
        pass

    @abstractmethod
    def sample(
        self,
        num_samples: int,
        *,
        class_idx: Optional[int] = None,
        **kwargs: Any,
    ) -> Tensor:
        pass

    @abstractmethod
    def interpolate(
        self,
        num_samples: int,
        *,
        class_idx: Optional[int] = None,
        use_slerp: bool = False,
        **kwargs: Any,
    ) -> Tensor:
        pass

    # utilities

    @property
    def is_conditional(self) -> bool:
        return self.num_classes is not None

    def reconstruct(self, net: Tensor, **kwargs: Any) -> Tensor:
        name = self.__class__.__name__
        if not self.can_reconstruct:
            raise ValueError(f"`{name}` does not support `reconstruct`")
        labels = kwargs.pop(LABEL_KEY, None)
        batch = {INPUT_KEY: net, LABEL_KEY: labels}
        if self.is_conditional and labels is None:
            raise ValueError(
                f"`{LABEL_KEY}` should be provided in `reconstruct` "
                f"for conditional `{name}`"
            )
        return _forward(self, 0, batch, INPUT_KEY, **kwargs)[PREDICTIONS_KEY]

    def get_sample_labels(
        self,
        num_samples: int,
        class_idx: Optional[int] = None,
    ) -> Optional[Tensor]:
        if self.num_classes is None:
            return None
        if class_idx is not None:
            return torch.full([num_samples], class_idx, device=get_device(self))
        return torch.randint(self.num_classes, [num_samples], device=get_device(self))


class GaussianGeneratorMixin(GeneratorMixin, metaclass=ABCMeta):
    @abstractmethod
    def decode(self, z: Tensor, *, labels: Optional[Tensor], **kwargs: Any) -> Tensor:
        pass

    def generate_z(self, num_samples: int) -> Tensor:
        return torch.randn(num_samples, self.latent_dim, device=get_device(self))

    def sample(
        self,
        num_samples: int,
        *,
        class_idx: Optional[int] = None,
        labels: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        z = self.generate_z(num_samples)
        if labels is None:
            labels = self.get_sample_labels(num_samples, class_idx)
        elif class_idx is not None:
            msg = "`class_idx` should not be provided when `labels` is provided"
            raise ValueError(msg)
        kw = shallow_copy_dict(kwargs)
        kw["z"] = z
        kw["labels"] = labels
        return safe_execute(self.decode, kw)

    def interpolate(
        self,
        num_samples: int,
        *,
        class_idx: Optional[int] = None,
        use_slerp: bool = False,
        **kwargs: Any,
    ) -> Tensor:
        z1 = self.generate_z(1)
        z2 = self.generate_z(1)
        shape = z1.shape
        z1 = z1.view(1, -1)
        z2 = z2.view(1, -1)
        ratio = torch.linspace(0.0, 1.0, num_samples, device=get_device(self))[:, None]
        z = slerp(z1, z2, ratio) if use_slerp else ratio * z1 + (1.0 - ratio) * z2
        z = z.view(num_samples, *shape[1:])
        if class_idx is None and self.num_classes is not None:
            class_idx = random.randint(0, self.num_classes - 1)
        kw = shallow_copy_dict(kwargs)
        kw["z"] = z
        kw["labels"] = self.get_sample_labels(num_samples, class_idx)
        return safe_execute(self.decode, kw)