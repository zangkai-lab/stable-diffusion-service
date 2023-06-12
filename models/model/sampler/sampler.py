import tqdm

from abc import abstractmethod
from typing import Dict, Any, Optional, Type
from torch import Tensor

from tools.bases.register import WithRegister
from tools.utils.safe import safe_execute
from tools.utils.icopy import shallow_copy_dict
from tools.utils.update import update_dict

from models.model.diffusion.utils import get_timesteps, cond_type
from models.model.diffusion.diffusion import IDiffusion

samplers: Dict[str, Type["ISampler"]] = {}


class IQSampler:
    def __init__(self, model: IDiffusion):
        self.model = model

    @abstractmethod
    def q_sample(
        self,
        net: Tensor,
        timesteps: Tensor,
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        pass

    @abstractmethod
    def reset_buffers(self, **kwargs: Any) -> None:
        pass


class ISampler(WithRegister):
    d = samplers

    default_steps: int

    def __init__(self, model: IDiffusion):
        self.model = model
        self.initialized = False

    @property
    @abstractmethod
    def q_sampler(self) -> IQSampler:
        pass

    @property
    @abstractmethod
    def sample_kwargs(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def sample_step(
        self,
        image: Tensor,
        cond: Optional[cond_type],
        step: int,
        total_step: int,
        **kwargs: Any,
    ) -> Tensor:
        pass

    def q_sample(
        self,
        net: Tensor,
        timesteps: Tensor,
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        return self.q_sampler.q_sample(net, timesteps, noise)

    def sample(
        self,
        z: Tensor,
        *,
        ref: Optional[Tensor] = None,
        ref_mask: Optional[Tensor] = None,
        ref_noise: Optional[Tensor] = None,
        cond: Optional[Any] = None,
        num_steps: Optional[int] = None,
        start_step: Optional[int] = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        # setup
        if num_steps is None:
            num_steps = getattr(self, "default_steps", self.model.t)
            assert isinstance(num_steps, int)
        if start_step is None:
            start_step = 0
        iterator = list(range(start_step, num_steps))
        if verbose:
            iterator = tqdm(iterator, desc=f"sampling ({self.__identifier__})")
        # execute
        image = z
        if cond is not None and self.model.condition_model is not None:
            cond = self.model._get_cond(cond)
        for step in iterator:
            callback = kwargs.get("step_callback")
            if callback is not None:
                callback_kw = dict(step=step, num_steps=num_steps, image=image)
                if not safe_execute(callback, callback_kw):
                    break
            kw = shallow_copy_dict(self.sample_kwargs)
            update_dict(shallow_copy_dict(kwargs), kw)
            image = self.sample_step(image, cond, step, num_steps, **kw)
            if ref is not None and ref_mask is not None and ref_noise is not None:
                ref_ts = get_timesteps(num_steps - step - 1, ref.shape[0], z.device)
                ref_noisy = self.q_sample(ref, ref_ts, ref_noise)
                image = ref_noisy * ref_mask + image * (1.0 - ref_mask)
        self.initialized = False
        return image
