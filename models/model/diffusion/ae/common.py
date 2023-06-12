import torch
import torch.nn as nn

from abc import abstractmethod, ABCMeta
from typing import Optional, List, Dict, Any, Callable, Tuple
from torch import Tensor

from tools.utils.type import tensor_dict_type
from tools.utils.device import get_device

from models.model.diffusion.generator import GaussianGeneratorMixin
from models.model.custom import ModelWithCustomSteps, CustomTrainStepLoss
from models.model.metrics import MetricsOutputs


class LPIPS(nn.Module):
    def __init__(self, use_dropout: bool = True):
        from ..models.cv.encoder.backbone.core import Backbone

        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.backbone = Backbone("vgg16_full", pretrained=True, requires_grad=False)
        make_mapping = lambda in_nc: ChannelMapping(in_nc, use_dropout)
        self.out_channels = self.backbone.out_channels
        self.mappings = nn.ModuleList(list(map(make_mapping, self.out_channels)))
        self.load_pretrained()
        set_requires_grad(self, False)

    def load_pretrained(self) -> None:
        ckpt = download_model("lpips")
        self.load_state_dict(torch.load(ckpt), strict=False)

    def forward(self, predictions: Tensor, target: Tensor) -> Tensor:
        net0, net1 = map(self.scaling_layer, [predictions, target])
        out0, out1 = map(self.backbone, [net0, net1])
        loss = None
        for i in range(len(self.out_channels)):
            stage = f"stage{i}"
            f0, f1 = out0[stage], out1[stage]
            f0, f1 = map(normalize, [f0, f1])
            diff = (f0 - f1) ** 2
            squeezed = self.mappings[i](diff)
            i_loss = spatial_average(squeezed, keepdim=True)
            if loss is None:
                loss = i_loss
            else:
                loss += i_loss
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