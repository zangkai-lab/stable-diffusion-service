import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import abstractmethod, ABCMeta
from typing import Optional, List, Dict, Any, Callable, Tuple, Union
from torch import Tensor
from torch.autograd import grad

from tools.utils.type import tensor_dict_type
from tools.utils.device import get_device

from models.model.diffusion.generator import GaussianGeneratorMixin
from models.model.custom import ModelWithCustomSteps, CustomTrainStepLoss, CustomTrainStep
from models.model.metrics import MetricsOutputs
from models.model.constant import INPUT_KEY
from models.model.train_state import TrainerState

from models.model.diffusion.ae.lpips import LPIPS
from models.model.discriminators.nlayer import NLayerDiscriminator


def d_hinge_loss(real: Tensor, fake: Tensor) -> Tensor:
    l_real = torch.mean(F.relu(1.0 - real))
    l_fake = torch.mean(F.relu(1.0 + fake))
    loss = 0.5 * (l_real + l_fake)
    return loss


def d_vanilla_loss(real: Tensor, fake: Tensor) -> Tensor:
    l_real = torch.mean(F.softplus(-real))
    l_fake = torch.mean(F.softplus(fake))
    loss = 0.5 * (l_real + l_fake)
    return loss


# 相似度判别模型
class AutoEncoderLPIPSWithDiscriminator(nn.Module, metaclass=ABCMeta):
    def __init__(
        self,
        *,
        kl_weight: float = 1.0,
        d_loss: str = "hinge",
        d_loss_start_step: int = 50001,
        d_num_layers: int = 4,
        d_in_channels: int = 3,
        d_start_channels: int = 64,
        d_factor: float = 1.0,
        d_weight: float = 1.0,
        perceptual_weight: float = 1.0,
    ):
        super().__init__()
        assert d_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.discriminator = NLayerDiscriminator(
            in_channels=d_in_channels,
            num_layers=d_num_layers,
            start_channels=d_start_channels,
        )
        self.d_loss_start_step = d_loss_start_step
        self.d_loss = d_hinge_loss if d_loss == "hinge" else d_vanilla_loss
        self.d_factor = d_factor
        self.d_weight = d_weight

    def get_d_weight(
        self,
        nll_loss: Tensor,
        g_loss: Tensor,
        last_layer: nn.Parameter,
    ) -> Tensor:
        nll_grads = grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = grad(g_loss, last_layer, retain_graph=True)[0]
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1.0e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1.0e4).detach()
        d_weight = d_weight * self.d_weight
        return d_weight

    def g_loss(
        self,
        nll_loss: Tensor,
        last_layer: nn.Parameter,
        loss_items: Dict[str, float],
        reconstructions: Tensor,
        cond: Optional[Tensor],
    ) -> Tensor:
        device = nll_loss.device
        if cond is None:
            fake = self.discriminator(reconstructions).output
        else:
            fake = self.discriminator(torch.cat((reconstructions, cond), dim=1)).output
        g_loss = -torch.mean(fake)
        if self.d_factor <= 0.0:
            d_weight = torch.tensor(0.0, device=device)
        else:
            try:
                d_weight = self.get_d_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0, device=device)
        loss_items["g_loss"] = g_loss.item()
        return d_weight * self.d_factor * g_loss

    def get_discriminator_loss(
        self,
        inputs: Tensor,
        reconstructions: Tensor,
        *,
        step: Optional[int],
        cond: Optional[Tensor] = None,
    ) -> CustomTrainStepLoss:
        if step is not None and step < self.d_loss_start_step:
            raise ValueError(
                "should not call `get_discriminator_loss` because current step "
                f"({step}) is smaller than the `d_loss_start_step` "
                f"({self.d_loss_start_step})"
            )
        inputs = inputs.contiguous().detach()
        reconstructions = reconstructions.contiguous().detach()
        if cond is None:
            real = self.discriminator(inputs).output
            fake = self.discriminator(reconstructions).output
        else:
            real = self.discriminator(torch.cat((inputs, cond), dim=1)).output
            fake = self.discriminator(torch.cat((reconstructions, cond), dim=1)).output
        loss = self.d_factor * self.d_loss(real, fake)
        return CustomTrainStepLoss(loss, {"d_loss": loss.item()})

    @abstractmethod
    def get_generator_loss(
        self,
        batch: tensor_dict_type,
        forward_results: tensor_dict_type,
        *,
        step: Optional[int],
        last_layer: nn.Parameter,
        cond: Optional[Tensor] = None,
    ) -> CustomTrainStepLoss:
        pass


err_fmt = "`loss` is not initialized for `{}`"


class GeneratorStep(CustomTrainStep):
    def loss_fn(
        self,
        m: "IAutoEncoder",
        state: Optional[TrainerState],
        batch: tensor_dict_type,
        forward_results: Union[tensor_dict_type, List[tensor_dict_type]],
        **kwargs: Any,
    ) -> CustomTrainStepLoss:
        if m.loss is None:
            raise ValueError(err_fmt.format(m.__class__.__name__))
        return m.loss.get_generator_loss(
            batch,
            forward_results,
            step=None if state is None else state.step,
            last_layer=m.generator.decoder.head[-1].weight,  # type: ignore
        )


class DiscriminatorStep(CustomTrainStep):
    def should_skip(self, m: "IAutoEncoder", state: TrainerState) -> bool:
        if m.loss is None:
            raise ValueError(err_fmt.format(m.__class__.__name__))
        return state.step < m.loss.d_loss_start_step

    def loss_fn(
        self,
        m: "IAutoEncoder",
        state: Optional["TrainerState"],
        batch: tensor_dict_type,
        forward_results: Union[tensor_dict_type, List[tensor_dict_type]],
        **kwargs: Any,
    ) -> CustomTrainStepLoss:
        if m.loss is None:
            raise ValueError(err_fmt.format(m.__class__.__name__))
        return m.loss.get_discriminator_loss(
            batch[INPUT_KEY],
            forward_results[PREDICTIONS_KEY],  # type: ignore
            step=None if state is None else state.step,
        )


class IAutoEncoder(GaussianGeneratorMixin, ModelWithCustomSteps, metaclass=ABCMeta):
    loss: Optional[AutoEncoderLPIPSWithDiscriminator]
    generator: nn.Module
    to_embedding: nn.Module
    from_embedding: nn.Module

    z_size: int
    img_size: int
    grad_accumulate: int
    embedding_channels: int

    @property
    def use_loss(self) -> bool:
        return self.loss is not None

    @property
    def ae_parameters(self) -> List[nn.Parameter]:
        return (
            list(self.generator.parameters())
            + list(self.to_embedding.parameters())
            + list(self.from_embedding.parameters())
        )

    @property
    def d_parameters(self) -> List[nn.Parameter]:
        if self.loss is None:
            return []
        return list(self.loss.discriminator.parameters())

    @property
    def can_reconstruct(self) -> bool:
        return True

    @property
    def train_steps(self) -> List[CustomTrainStep]:
        g_scope = "core.ae_parameters"
        d_scope = "core.d_parameters"
        return [
            GeneratorStep(g_scope, grad_accumulate=self.grad_accumulate),
            DiscriminatorStep(
                d_scope,
                grad_accumulate=self.grad_accumulate,
                requires_new_forward=True,
                requires_grad_in_forward=True,
            ),
        ]

    def evaluate(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState],
        weighted_loss_score_fn: Callable[[Dict[str, float]], float],
        forward_kwargs: Dict[str, Any],
    ) -> MetricsOutputs:
        forward = self.forward(batch[INPUT_KEY])
        args = self, state, batch, forward
        loss_items = {}
        g_out = GeneratorStep().loss_fn(*args)
        loss_items.update(g_out.losses)
        if self.loss is None:
            raise ValueError(err_fmt.format(self.__class__.__name__))
        if state is None or state.step >= self.loss.d_loss_start_step:
            d_out = DiscriminatorStep().loss_fn(*args)
            loss_items.update(d_out.losses)
        score = -loss_items["recon"]
        return MetricsOutputs(score, loss_items, {k: False for k in loss_items})

    def generate_z(self, num_samples: int) -> Tensor:
        z = torch.randn(num_samples, self.embedding_channels, self.z_size, self.z_size)
        return z.to(get_device(self))

    def setup(
        self,
        img_size: int,
        grad_accumulate: int,
        embedding_channels: int,
        channel_multipliers: Tuple[int, ...],
    ) -> None:
        self.z_size = img_size // 2 ** len(channel_multipliers)
        self.embedding_channels = embedding_channels
        self.grad_accumulate = grad_accumulate

    @abstractmethod
    def forward(self, net: Tensor) -> tensor_dict_type:
        pass