import numpy as np

from typing import Optional, Tuple, Dict, Any
from torch import Tensor

from models.model.model_dl import IDLModel
from models.model.diffusion.ddpm import DDPM, make_condition_model
from models.zoo.core import DLZoo
from models.model.diffusion.utils import CROSS_ATTN_TYPE, freeze

from models.model.diffusion.ae.kl import GaussianDistribution


@IDLModel.register("ldm")
class LDM(DDPM):
    def __init__(
        self,
        img_size: int,
        # unet
        in_channels: int,
        out_channels: int,
        *,
        num_heads: Optional[int] = 8,
        num_head_channels: Optional[int] = None,
        use_spatial_transformer: bool = False,
        num_transformer_layers: int = 1,
        context_dim: Optional[int] = None,
        signal_dim: int = 2,
        start_channels: int = 320,
        num_res_blocks: int = 2,
        attention_downsample_rates: Tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.0,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
        resample_with_conv: bool = True,
        resample_with_resblock: bool = False,
        use_scale_shift_norm: bool = False,
        num_classes: Optional[int] = None,
        use_linear_in_transformer: bool = False,
        use_checkpoint: bool = False,
        attn_split_chunk: Optional[int] = None,
        tome_info: Optional[Dict[str, Any]] = None,
        # first stage
        first_stage: str,
        first_stage_config: Optional[Dict[str, Any]] = None,
        first_stage_scale_factor: float = 1.0,
        # diffusion
        ema_decay: Optional[float] = None,
        use_num_updates_in_ema: bool = True,
        parameterization: str = "eps",
        ## condition
        condition_type: str = CROSS_ATTN_TYPE,
        condition_model: Optional[str] = None,
        use_first_stage_as_condition: bool = False,
        condition_config: Optional[Dict[str, Any]] = None,
        condition_learnable: bool = False,
        use_pretrained_condition: bool = False,
        ## noise schedule
        v_posterior: float = 0.0,
        timesteps: int = 1000,
        given_betas: Optional[np.ndarray] = None,
        beta_schedule: str = "linear",
        linear_start: float = 1.0e-4,
        linear_end: float = 2.0e-2,
        cosine_s: float = 8.0e-3,
        ## loss
        loss_type: str = "l2",
        l_simple_weight: float = 1.0,
        original_elbo_weight: float = 0.0,
        learn_log_var: bool = False,
        log_var_init: float = 0.0,
        ## sampling
        sampler: str = "ddim",
        sampler_config: Optional[Dict[str, Any]] = None,
    ):
        self.use_first_stage_as_condition = use_first_stage_as_condition
        if use_first_stage_as_condition:
            if condition_learnable:
                raise ValueError(
                    "should not use ae as condition model "
                    "when `condition_learnable` is set to True"
                )
            condition_model = None
        super().__init__(
            img_size,
            in_channels,
            out_channels,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            use_spatial_transformer=use_spatial_transformer,
            num_transformer_layers=num_transformer_layers,
            context_dim=context_dim,
            signal_dim=signal_dim,
            start_channels=start_channels,
            num_res_blocks=num_res_blocks,
            attention_downsample_rates=attention_downsample_rates,
            dropout=dropout,
            channel_multipliers=channel_multipliers,
            resample_with_conv=resample_with_conv,
            resample_with_resblock=resample_with_resblock,
            use_scale_shift_norm=use_scale_shift_norm,
            num_classes=num_classes,
            use_linear_in_transformer=use_linear_in_transformer,
            use_checkpoint=use_checkpoint,
            attn_split_chunk=attn_split_chunk,
            tome_info=tome_info,
            ema_decay=ema_decay,
            use_num_updates_in_ema=use_num_updates_in_ema,
            parameterization=parameterization,
            condition_type=condition_type,
            condition_model=condition_model,
            condition_config=condition_config,
            condition_learnable=condition_learnable,
            use_pretrained_condition=use_pretrained_condition,
            v_posterior=v_posterior,
            timesteps=timesteps,
            given_betas=given_betas,
            beta_schedule=beta_schedule,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
            loss_type=loss_type,
            l_simple_weight=l_simple_weight,
            original_elbo_weight=original_elbo_weight,
            learn_log_var=learn_log_var,
            log_var_init=log_var_init,
            sampler=sampler,
            sampler_config=sampler_config,
        )
        first_stage_kw = first_stage_config or {}
        m = DLZoo.load_pipeline(first_stage, **first_stage_kw)
        self.first_stage = freeze(m.build_model.model)
        self.scale_factor = first_stage_scale_factor
        # condition
        if use_first_stage_as_condition:
            # avoid recording duplicate parameters
            self.condition_model = [make_condition_model(first_stage, self.first_stage)]
        # sanity check
        embedding_channels = self.first_stage.embedding_channels
        if in_channels != embedding_channels and condition_type == CROSS_ATTN_TYPE:
            raise ValueError(
                f"`in_channels` ({in_channels}) should be identical with the "
                f"`embedding_channels` ({embedding_channels}) of the "
                f"first_stage model ({first_stage})"
            )
        if out_channels != embedding_channels:
            raise ValueError(
                f"`out_channels` ({out_channels}) should be identical with the "
                f"`embedding_channels` ({embedding_channels}) of the "
                f"first_stage model ({first_stage})"
            )

    def decode(
        self,
        z: Tensor,
        *,
        cond: Optional[Any] = None,
        num_steps: Optional[int] = None,
        start_step: Optional[int] = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        latent = super().decode(
            z,
            cond=cond,
            num_steps=num_steps,
            start_step=start_step,
            verbose=verbose,
            **kwargs,
        )
        if kwargs.get("return_latent", False):
            return latent
        callback = kwargs.get("decode_callback")
        if callback is not None:
            callback()
        net = self._from_latent(latent)
        return net

    def _preprocess(self, net: Tensor, *, deterministic: bool = False) -> Tensor:
        net = self.first_stage.encode(net)
        if isinstance(net, GaussianDistribution):
            net = net.mode() if deterministic else net.sample()
        net = self.scale_factor * net
        return net

    def _from_latent(self, latent: Tensor) -> Tensor:
        latent = latent / self.scale_factor
        return self.first_stage.decode(latent, resize=False)

    def _get_cond(self, cond: Any) -> Tensor:
        if isinstance(self.condition_model, list):
            return self.condition_model[0](cond)
        return super()._get_cond(cond)