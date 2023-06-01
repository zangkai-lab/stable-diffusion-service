import os
import torch
import torch.nn as nn
import numpy as np

from typing import Any, Dict, List, Optional, Tuple, Union

from models.pipeline.pipeline_model import ModelPipeline
from models.pooling.pool_weights import WeightsPool


class DiffusionAPI(ModelPipeline):
    m: DDPM
    sampler: ISampler
    cond_model: Optional[nn.Module]
    first_stage: Optional[IAutoEncoder]
    latest_seed: int
    latest_variation_seed: Optional[int]
    sd_weights: WeightsPool
    current_sd_version: Optional[str]

    def __init__(
        self,
        m: DDPM,
        device: torch.device,
        *,
        use_amp: bool = False,
        use_half: bool = False,
        clip_skip: int = 0,
    ):
        super().__init__(m, device, use_amp=use_amp, use_half=use_half)
        self.sampler = m.sampler
        self.cond_type = m.condition_type
        self.clip_skip = clip_skip
        self.sd_weights = WeightsPool()
        self.current_sd_version = None
        # extracted the condition models so we can pre-calculate the conditions
        self.cond_model = m.condition_model
        if self.cond_model is not None:
            self.cond_model.eval()
        m.condition_model = nn.Identity()
        # pre-calculate unconditional_cond if needed
        self._original_raw_uncond = getattr(m.sampler, "unconditional_cond", None)
        self._uncond_cache: tensor_dict_type = {}
        self._update_sampler_uncond(clip_skip)
        # extract first stage
        if not isinstance(m, LDM):
            self.first_stage = None
        else:
            self.first_stage = m.first_stage

    # api

    @property
    def size_info(self) -> SizeInfo:
        opt_size = self.m.img_size
        if self.first_stage is None:
            factor = 1
        else:
            factor = self.first_stage.img_size // opt_size
        return SizeInfo(factor, opt_size)

    def to(
        self,
        device: torch.device,
        *,
        use_amp: bool = False,
        use_half: bool = False,
    ) -> None:
        super().to(device, use_amp=use_amp, use_half=use_half)
        unconditional_cond = getattr(self.sampler, "unconditional_cond", None)
        if use_half:
            if self.cond_model is not None:
                self.cond_model.half()
            if self.first_stage is not None:
                self.first_stage.half()
            if unconditional_cond is not None:
                unconditional_cond = unconditional_cond.half()
            for k, v in self._uncond_cache.items():
                self._uncond_cache[k] = v.half()
        else:
            if self.cond_model is not None:
                self.cond_model.float()
            if self.first_stage is not None:
                self.first_stage.float()
            if unconditional_cond is not None:
                unconditional_cond = unconditional_cond.float()
            for k, v in self._uncond_cache.items():
                self._uncond_cache[k] = v.float()
        if self.cond_model is not None:
            self.cond_model.to(device)
        if self.first_stage is not None:
            self.first_stage.to(device)
        if unconditional_cond is not None:
            self.sampler.unconditional_cond = unconditional_cond.to(device)
        for k, v in self._uncond_cache.items():
            self._uncond_cache[k] = v.to(device)

    def prepare_sd(
        self,
        versions: List[str],
        *,
        # inpainting workarounds
        # should set `force_external` to `True` to prepare inpainting with this method
        sub_folder: Optional[str] = None,
        force_external: bool = False,
    ) -> None:
        root = os.path.join(OPT.cache_dir, DLZoo.model_dir)
        for tag in map(get_sd_tag, versions):
            if tag not in self.sd_weights:
                _load_external = lambda: _convert_external(self, tag, sub_folder)
                if force_external:
                    model_path = _load_external()
                else:
                    try:
                        model_path = download_model(f"ldm_sd_{tag}", root=root)
                    except:
                        model_path = _load_external()
                self.sd_weights.register(tag, model_path)

    def switch_sd(self, version: str) -> None:
        tag = get_sd_tag(version)
        if self.current_sd_version is not None:
            if tag == get_sd_tag(self.current_sd_version):
                return
        d = self.sd_weights.get(tag)
        with self.load_context() as m:
            m.load_state_dict(d)
        self.current_sd_version = version

    def get_cond(self, cond: Any) -> Tensor:
        if self.cond_model is None:
            msg = "should not call `get_cond` when `cond_model` is not available"
            raise ValueError(msg)
        with torch.no_grad():
            with self.amp_context:
                return self.cond_model(cond)

    def switch_sampler(
        self,
        sampler: str,
        sampler_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        sampler_ins = self.m.make_sampler(sampler, sampler_config)
        current_unconditional_cond = getattr(self.m.sampler, "unconditional_cond", None)
        if current_unconditional_cond is not None:
            if hasattr(sampler_ins, "unconditional_cond"):
                sampler_ins.unconditional_cond = current_unconditional_cond
        current_guidance = getattr(self.m.sampler, "unconditional_guidance_scale", None)
        if current_guidance is not None:
            if hasattr(sampler_ins, "unconditional_guidance_scale"):
                sampler_ins.unconditional_guidance_scale = current_guidance
        self.sampler = self.m.sampler = sampler_ins

    def switch_circular(self, enable: bool) -> None:
        def _inject(m: nn.Module) -> None:
            for child in m.children():
                _inject(child)
            modules.append(m)

        padding_mode = "circular" if enable else "zeros"
        modules: List[nn.Module] = []
        _inject(self.m)
        for module in modules:
            if isinstance(module, nn.Conv2d):
                module.padding_mode = padding_mode
            elif isinstance(module, Conv2d):
                module.padding = padding_mode

    def sample(
        self,
        num_samples: int,
        export_path: Optional[str] = None,
        *,
        seed: Optional[int] = None,
        use_seed_resize: bool = False,
        # each variation contains (seed, weight)
        variations: Optional[List[Tuple[int, float]]] = None,
        variation_seed: Optional[int] = None,
        variation_strength: Optional[float] = None,
        z: Optional[Tensor] = None,
        z_ref: Optional[Tensor] = None,
        z_ref_mask: Optional[Tensor] = None,
        z_ref_noise: Optional[Tensor] = None,
        size: Optional[Tuple[int, int]] = None,
        original_size: Optional[Tuple[int, int]] = None,
        alpha: Optional[np.ndarray] = None,
        cond: Optional[Any] = None,
        cond_concat: Optional[Tensor] = None,
        unconditional_cond: Optional[Any] = None,
        hint: Optional[Union[Tensor, tensor_dict_type]] = None,
        hint_start: Optional[Union[float, Dict[str, float]]] = None,
        num_steps: Optional[int] = None,
        clip_output: bool = True,
        callback: Optional[Callable[[Tensor], Tensor]] = None,
        batch_size: Optional[int] = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        o_kw_backup = dict(
            seed=seed,
            variations=variations,
            alpha=alpha,
            cond=cond,
            cond_concat=cond_concat,
            unconditional_cond=unconditional_cond,
            hint=hint,
            hint_start=hint_start,
            num_steps=num_steps,
            clip_output=clip_output,
            callback=callback,
            batch_size=batch_size,
            verbose=verbose,
        )
        if batch_size is None:
            batch_size = num_samples
        registered_custom = False
        if self.cond_model is not None:
            clip_skip = kwargs.get(
                "clip_skip",
                0 if self.clip_skip is None else self.clip_skip,
            )
            self._update_sampler_uncond(clip_skip)
            if isinstance(self.cond_model, CLIPTextConditionModel):
                custom_embeddings = kwargs.get("custom_embeddings")
                if custom_embeddings is not None:
                    registered_custom = True
                    self.cond_model.register_custom(custom_embeddings)
        if cond is not None:
            if self.cond_type != CONCAT_TYPE and self.cond_model is not None:
                cond = predict_array_data(
                    self.cond_model,
                    ArrayData.init().fit(np.array(cond)),
                    batch_size=batch_size,
                )[PREDICTIONS_KEY]
        if cond is not None and num_samples != len(cond):
            raise ValueError(
                f"`num_samples` ({num_samples}) should be identical with "
                f"the number of `cond` ({len(cond)})"
            )
        if alpha is not None and original_size is not None:
            alpha_h, alpha_w = alpha.shape[-2:]
            if alpha_w != original_size[0] or alpha_h != original_size[1]:
                raise ValueError(
                    f"shape of the provided `alpha` ({alpha_w}, {alpha_h}) should be "
                    f"identical with the provided `original_size` {original_size}"
                )
        unconditional = cond is None
        if unconditional:
            cond = [0] * num_samples
        cond_data: ArrayData = ArrayData.init(DataConfig(batch_size=batch_size))
        cond_data.fit(cond)
        iterator = TensorBatcher(cond_data.get_loaders()[0], self.device)
        num_iter = len(iterator)
        if verbose and num_iter > 1:
            iterator = tqdm(iterator, desc="iter", total=num_iter)
        sampled = []
        kw = dict(num_steps=num_steps, verbose=verbose)
        kw.update(shallow_copy_dict(kwargs))
        factor, opt_size = self.size_info
        if size is None:
            size = opt_size, opt_size
        else:
            size = tuple(map(lambda n: round(n / factor), size))  # type: ignore
        uncond_backup = None
        unconditional_cond_backup = None
        if self.cond_model is not None and unconditional_cond is not None:
            uncond_backup = getattr(self.sampler, "uncond", None)
            unconditional_cond_backup = getattr(
                self.sampler,
                "unconditional_cond",
                None,
            )
            uncond = self.get_cond(unconditional_cond).to(self.device)
            self.sampler.uncond = uncond.clone()
            self.sampler.unconditional_cond = uncond.clone()
        highres_info = kwargs.get("highres_info")
        with eval_context(self.m):
            with self.amp_context:
                for i, batch in enumerate(iterator):
                    # from the 2nd batch forward, we need to re-generate new seeds
                    if i >= 1:
                        seed = new_seed()
                    i_kw = shallow_copy_dict(kw)
                    i_kw_backup = shallow_copy_dict(i_kw)
                    i_cond = batch[INPUT_KEY].to(self.device)
                    i_n = len(i_cond)
                    repeat = (
                        lambda t: t
                        if t.shape[0] == i_n
                        else t.repeat_interleave(i_n, dim=0)
                    )
                    if z is not None:
                        i_z = repeat(z)
                    else:
                        in_channels = self.m.in_channels
                        if self.cond_type == CONCAT_TYPE:
                            in_channels -= cond.shape[1]
                        elif cond_concat is not None:
                            in_channels -= cond_concat.shape[1]
                        i_z_shape = i_n, in_channels, *size[::-1]
                        i_z, _ = self._set_seed_and_variations(
                            seed,
                            lambda: torch.randn(i_z_shape, device=self.device),
                            lambda noise: noise,
                            variations,
                            variation_seed,
                            variation_strength,
                        )
                    if use_seed_resize:
                        z_original_shape = list(i_z.shape[-2:])
                        z_opt_shape = list(
                            map(lambda n: round(n / factor), [opt_size, opt_size])
                        )
                        if z_original_shape != z_opt_shape:
                            dx = (z_original_shape[0] - z_opt_shape[0]) // 2
                            dy = (z_original_shape[1] - z_opt_shape[1]) // 2
                            x = z_opt_shape[0] if dx >= 0 else z_opt_shape[0] + 2 * dx
                            y = z_opt_shape[1] if dy >= 0 else z_opt_shape[1] + 2 * dy
                            dx = max(-dx, 0)
                            dy = max(-dy, 0)
                            i_opt_z_shape = (
                                i_n,
                                self.m.in_channels,
                                *z_opt_shape,
                            )
                            i_opt_z, _ = self._set_seed_and_variations(
                                seed,
                                lambda: torch.randn(i_opt_z_shape, device=self.device),
                                lambda noise: noise,
                                variations,
                                variation_seed,
                                variation_strength,
                            )
                            i_z[..., dx : dx + x, dy : dy + y] = i_opt_z[
                                ..., dx : dx + x, dy : dy + y
                            ]
                    if z_ref is not None and z_ref_mask is not None:
                        if z_ref_noise is not None:
                            i_kw["ref"] = repeat(z_ref)
                            i_kw["ref_mask"] = repeat(z_ref_mask)
                            i_kw["ref_noise"] = repeat(z_ref_noise)
                    if unconditional:
                        i_cond = None
                    if self.use_half:
                        i_z = i_z.half()
                        if i_cond is not None:
                            i_cond = i_cond.half()
                        for k, v in i_kw.items():
                            if isinstance(v, torch.Tensor) and v.is_floating_point():
                                i_kw[k] = v.half()
                    if cond_concat is not None:
                        if self.cond_type != HYBRID_TYPE:
                            raise ValueError(
                                f"condition type should be `{HYBRID_TYPE}` when "
                                f"`cond_concat` is provided"
                            )
                        i_cond = {
                            CROSS_ATTN_KEY: i_cond,
                            CONCAT_KEY: cond_concat,
                        }
                    if hint is not None:
                        if isinstance(i_cond, dict):
                            i_cond[CONTROL_HINT_KEY] = hint
                            i_cond[CONTROL_HINT_START_KEY] = hint_start
                        else:
                            i_cond = {
                                CROSS_ATTN_KEY: i_cond,
                                CONTROL_HINT_KEY: hint,
                                CONTROL_HINT_START_KEY: hint_start,
                            }
                    with switch_sampler_context(self, i_kw.get("sampler")):
                        if highres_info is not None:
                            # highres workaround
                            i_kw["return_latent"] = True
                        i_sampled = self.m.decode(i_z, cond=i_cond, **i_kw)
                        if highres_info is not None:
                            i_z = self._get_highres_latent(i_sampled, highres_info)
                            fidelity = highres_info["fidelity"]
                            if num_steps is None:
                                num_steps = self.sampler.default_steps
                            i_num_steps = get_highres_steps(num_steps, fidelity)
                            i_kw_backup.pop("highres_info", None)
                            i_kw_backup.update(o_kw_backup)
                            i_kw_backup["fidelity"] = fidelity
                            i_kw_backup["num_steps"] = i_num_steps
                            i_kw_backup["decode_callback"] = self.empty_cuda_cache
                            self.empty_cuda_cache()
                            i_sampled = self._img2img(i_z, export_path, **i_kw_backup)
                            if original_size is not None:
                                upscale_factor = highres_info["upscale_factor"]
                                original_size = (
                                    round(original_size[0] * upscale_factor),
                                    round(original_size[1] * upscale_factor),
                                )
                    sampled.append(i_sampled.cpu().float())
        if uncond_backup is not None:
            self.sampler.uncond = uncond_backup
        if unconditional_cond_backup is not None:
            self.sampler.unconditional_cond = unconditional_cond_backup
        concat = torch.cat(sampled, dim=0)
        if clip_output:
            concat = torch.clip(concat, -1.0, 1.0)
        if callback is not None:
            concat = callback(concat)
        if original_size is not None:
            original_size = (max(original_size[0], 1), max(original_size[1], 1))
            with torch.no_grad():
                concat = F.interpolate(
                    concat,
                    original_size[::-1],
                    mode="bicubic",
                )
        if alpha is not None:
            alpha = torch.from_numpy(2.0 * alpha - 1.0)
            if original_size is None:
                with torch.no_grad():
                    alpha = F.interpolate(
                        alpha,
                        concat.shape[-2:],
                        mode="nearest",
                    )
            concat = torch.cat([concat, alpha], dim=1)
        if export_path is not None:
            save_images(concat, export_path)
        self.empty_cuda_cache()
        if registered_custom:
            self.cond_model.clear_custom()
        return concat

    def txt2img(
        self,
        txt: Union[str, List[str]],
        export_path: Optional[str] = None,
        *,
        anchor: int = 64,
        max_wh: int = 512,
        num_samples: Optional[int] = None,
        size: Optional[Tuple[int, int]] = None,
        num_steps: Optional[int] = None,
        clip_output: bool = True,
        callback: Optional[Callable[[Tensor], Tensor]] = None,
        batch_size: Optional[int] = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        txt, num_samples = get_txt_cond(txt, num_samples)
        new_size = get_size(size, anchor, max_wh)
        return self.sample(
            num_samples,
            export_path,
            size=new_size,
            original_size=size,
            cond=txt,
            num_steps=num_steps,
            clip_output=clip_output,
            callback=callback,
            batch_size=batch_size,
            verbose=verbose,
            **kwargs,
        )

    def txt2img_inpainting(
        self,
        txt: Union[str, List[str]],
        image: Union[str, Image.Image],
        mask: Union[str, Image.Image],
        export_path: Optional[str] = None,
        *,
        seed: Optional[int] = None,
        anchor: int = 64,
        max_wh: int = 512,
        num_samples: Optional[int] = None,
        num_steps: Optional[int] = None,
        clip_output: bool = True,
        keep_original: bool = False,
        use_raw_inpainting: bool = False,
        inpainting_settings: Optional[InpaintingSettings] = None,
        callback: Optional[Callable[[Tensor], Tensor]] = None,
        use_background_guidance: bool = False,
        use_reference: bool = False,
        reference_fidelity: float = 0.2,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        def get_z_ref_pack(
            normalized_image_: np.ndarray,
            normalized_mask_: np.ndarray,
        ) -> Tuple[Tensor, Tensor, Tensor]:
            z_ref = self._get_z(normalized_image_)
            z_ref_mask = 1.0 - F.interpolate(
                torch.from_numpy(normalized_mask_).to(z_ref),
                z_ref.shape[-2:],
                mode="bicubic",
            )
            if seed is not None:
                seed_everything(seed)
            z_ref_noise = torch.randn_like(z_ref)
            return z_ref, z_ref_mask, z_ref_noise

        def get_z_info_from(
            z_ref_: Optional[Tensor],
            fidelity_: float,
            shape_: Tuple[int, int],
        ) -> Tuple[Optional[Tensor], Optional[Tuple[int, int]]]:
            if z_ref_ is None:
                z = None
                size = tuple(map(lambda n: n * self.size_info.factor, shape_))
            else:
                size = None
                args = z_ref_, num_steps, fidelity_, seed
                z, _, start_step = self._q_sample(*args, **kwargs)
                kwargs["start_step"] = start_step
            return z, size  # type: ignore

        def paste_original(
            original_: Image.Image,
            mask_: Image.Image,
            sampled_: Tensor,
        ) -> Tensor:
            rgb = to_rgb(original_)
            rgb_normalized = normalize_image_to_diffusion(rgb)
            rgb_normalized = rgb_normalized.transpose([2, 0, 1])[None]
            mask_res_ = read_image(mask_, None, anchor=None, to_mask=True)
            remained_mask_ = ~(mask_res_.image >= 0.5)
            pasted = np.where(remained_mask_, rgb_normalized, sampled_.numpy())
            return torch.from_numpy(pasted)

        if inpainting_settings is None:
            inpainting_settings = InpaintingSettings()
        txt_list, num_samples = get_txt_cond(txt, num_samples)

        # raw inpainting
        if use_raw_inpainting:
            image_res = read_image(image, max_wh, anchor=anchor)
            mask_res = read_image(mask, max_wh, anchor=anchor, to_mask=True)
            cropped_res = get_cropped(image_res, mask_res, inpainting_settings)
            z_ref_pack = get_z_ref_pack(cropped_res.image, cropped_res.mask)
            z_ref, z_ref_mask, z_ref_noise = z_ref_pack
            z, size = get_z_info_from(
                z_ref if use_reference else None,
                reference_fidelity,
                z_ref.shape[-2:][::-1],
            )
            kw = shallow_copy_dict(kwargs)
            kw.update(
                dict(
                    z=z,
                    size=size,
                    export_path=export_path,
                    z_ref=z_ref,
                    z_ref_mask=z_ref_mask,
                    z_ref_noise=z_ref_noise,
                    original_size=image_res.original_size,
                    alpha=None,
                    cond=txt_list,
                    num_steps=num_steps,
                    clip_output=clip_output,
                    verbose=verbose,
                )
            )
            crop_controlnet(kw, cropped_res.crop_res)
            sampled = self.sample(num_samples, **kw)
            crop_res = cropped_res.crop_res
            if crop_res is not None:
                sampled = recover_with(
                    image_res.original,
                    sampled,
                    crop_res,
                    cropped_res.wh_ratio,
                    inpainting_settings,
                )
            if keep_original:
                original = image_res.original
                sampled = paste_original(original, mask_res.original, sampled)
            return sampled

        # 'real' inpainting
        res = self._get_masked_cond(
            image,
            mask,
            max_wh,
            anchor,
            lambda remained_mask, img: np.where(remained_mask, img, 0.5),
            lambda bool_mask: torch.from_numpy(bool_mask),
            inpainting_settings,
        )
        # sampling
        with switch_sampler_context(self, kwargs.get("sampler")):
            # calculate `z_ref` stuffs based on `use_image_guidance`
            if not use_background_guidance:
                z_ref = z_ref_mask = z_ref_noise = None
            else:
                z_ref, z_ref_mask, z_ref_noise = get_z_ref_pack(res.image, res.mask)
            # calculate `z` based on `z_ref`, if needed
            z_shape = res.remained_image_cond.shape[-2:][::-1]
            if not use_reference:
                args = None, reference_fidelity, z_shape
            elif z_ref is not None:
                args = z_ref, reference_fidelity, z_shape
            else:
                args = self._get_z(res.image), reference_fidelity, z_shape
            z, size = get_z_info_from(*args)
            # adjust ControlNet parameters
            crop_controlnet(kwargs, res.crop_res)
            # core
            sampled = self.sample(
                num_samples,
                export_path,
                seed=seed,
                z=z,
                z_ref=z_ref,
                z_ref_mask=z_ref_mask,
                z_ref_noise=z_ref_noise,
                size=size,  # type: ignore
                original_size=res.original_size,
                alpha=None,
                cond=txt_list,
                cond_concat=torch.cat([res.mask_cond, res.remained_image_cond], dim=1),
                num_steps=num_steps,
                clip_output=clip_output,
                callback=callback,
                verbose=verbose,
                **kwargs,
            )
            if res.crop_res is not None:
                sampled = recover_with(
                    res.original_image,
                    sampled,
                    res.crop_res,
                    res.wh_ratio,
                    inpainting_settings,
                )
        if keep_original:
            sampled = paste_original(res.original_image, res.original_mask, sampled)
        return sampled

    def outpainting(
        self,
        txt: str,
        image: Union[str, Image.Image],
        export_path: Optional[str] = None,
        *,
        anchor: int = 64,
        max_wh: int = 512,
        num_steps: Optional[int] = None,
        clip_output: bool = True,
        keep_original: bool = False,
        callback: Optional[Callable[[Tensor], Tensor]] = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        if isinstance(image, str):
            image = Image.open(image)
        if image.mode != "RGBA":
            raise ValueError("`image` should be `RGBA` in outpainting")
        *rgb, alpha = image.split()
        mask = Image.fromarray(255 - np.array(alpha))
        image = Image.merge("RGB", rgb)
        return self.txt2img_inpainting(
            txt,
            image,
            mask,
            export_path,
            anchor=anchor,
            max_wh=max_wh,
            num_steps=num_steps,
            clip_output=clip_output,
            keep_original=keep_original,
            callback=callback,
            verbose=verbose,
            **kwargs,
        )

    def img2img(
        self,
        image: Union[str, Image.Image, Tensor],
        export_path: Optional[str] = None,
        *,
        anchor: int = 32,
        max_wh: int = 512,
        fidelity: float = 0.2,
        alpha: Optional[np.ndarray] = None,
        cond: Optional[Any] = None,
        num_steps: Optional[int] = None,
        clip_output: bool = True,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        if isinstance(image, Tensor):
            original_size = tuple(image.shape[-2:][::-1])
        else:
            res = read_image(image, max_wh, anchor=anchor)
            image = res.image
            original_size = res.original_size
            if alpha is None:
                alpha = res.alpha
        z = self._get_z(image)
        highres_info = kwargs.pop("highres_info", None)
        if highres_info is not None:
            z = self._get_highres_latent(z, highres_info)
            if num_steps is None:
                num_steps = self.sampler.default_steps
            num_steps = get_highres_steps(num_steps, fidelity)
            upscale_factor = highres_info["upscale_factor"]
            original_size = (
                round(original_size[0] * upscale_factor),
                round(original_size[1] * upscale_factor),
            )
            if alpha is not None:
                with torch.no_grad():
                    alpha = F.interpolate(
                        torch.from_numpy(alpha),
                        original_size[::-1],
                        mode="nearest",
                    ).numpy()
        return self._img2img(
            z,
            export_path,
            fidelity=fidelity,
            original_size=original_size,  # type: ignore
            alpha=alpha,
            cond=cond,
            num_steps=num_steps,
            clip_output=clip_output,
            verbose=verbose,
            **kwargs,
        )

    def inpainting(
        self,
        image: Union[str, Image.Image],
        mask: Union[str, Image.Image],
        export_path: Optional[str] = None,
        *,
        anchor: int = 32,
        max_wh: int = 512,
        alpha: Optional[np.ndarray] = None,
        refine_fidelity: Optional[float] = None,
        num_steps: Optional[int] = None,
        clip_output: bool = True,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        # inpainting callback, will not trigger in refine stage
        def callback(out: Tensor) -> Tensor:
            final = torch.from_numpy(res.remained_image.copy())
            final += 0.5 * (1.0 + out) * (1.0 - res.remained_mask)
            return 2.0 * final - 1.0

        res = self._get_masked_cond(
            image,
            mask,
            max_wh,
            anchor,
            lambda remained_mask, img: np.where(remained_mask, img, 0.0),
            lambda bool_mask: torch.where(torch.from_numpy(bool_mask), 1.0, -1.0),
        )
        cond = torch.cat([res.remained_image_cond, res.mask_cond], dim=1)
        size = self._get_identical_size_with(res.remained_image_cond)
        # refine with img2img
        if refine_fidelity is not None:
            z = self._get_z(res.image)
            return self._img2img(
                z,
                export_path,
                fidelity=refine_fidelity,
                original_size=res.original_size,
                alpha=res.image_alpha if alpha is None else alpha,
                cond=cond,
                num_steps=num_steps,
                clip_output=clip_output,
                verbose=verbose,
                **kwargs,
            )
        # sampling
        return self.sample(
            1,
            export_path,
            size=size,  # type: ignore
            original_size=res.original_size,
            alpha=res.image_alpha if alpha is None else alpha,
            cond=cond,
            num_steps=num_steps,
            clip_output=clip_output,
            callback=callback,
            verbose=verbose,
            **kwargs,
        )

    def sr(
        self,
        image: Union[str, Image.Image],
        export_path: Optional[str] = None,
        *,
        anchor: int = 8,
        max_wh: int = 512,
        alpha: Optional[np.ndarray] = None,
        num_steps: Optional[int] = None,
        clip_output: bool = True,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        if not isinstance(self.m, LDM):
            raise ValueError("`sr` is now only available for `LDM` models")
        factor = 2 ** (len(self.m.first_stage.core.channel_multipliers) - 1)
        res = read_image(image, round(max_wh / factor), anchor=anchor)
        wh_ratio = res.original_size[0] / res.original_size[1]
        zh, zw = res.image.shape[-2:]
        sr_size = (zw, zw / wh_ratio) if zw > zh else (zh * wh_ratio, zh)
        sr_size = tuple(map(lambda n: round(factor * n), sr_size))  # type: ignore
        cond = torch.from_numpy(2.0 * res.image - 1.0).to(self.device)
        size = self._get_identical_size_with(cond)
        return self.sample(
            1,
            export_path,
            size=size,
            original_size=sr_size,
            alpha=res.alpha if alpha is None else alpha,
            cond=cond,
            num_steps=num_steps,
            clip_output=clip_output,
            verbose=verbose,
            **kwargs,
        )

    def semantic2img(
        self,
        semantic: Union[str, Image.Image],
        export_path: Optional[str] = None,
        *,
        anchor: int = 16,
        max_wh: int = 512,
        alpha: Optional[np.ndarray] = None,
        num_steps: Optional[int] = None,
        clip_output: bool = True,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        err_fmt = "`{}` is needed for `semantic2img`"
        if self.cond_model is None:
            raise ValueError(err_fmt.format("cond_model"))
        in_channels = getattr(self.cond_model, "in_channels", None)
        if in_channels is None:
            raise ValueError(err_fmt.format("cond_model.in_channels"))
        factor = getattr(self.cond_model, "factor", None)
        if factor is None:
            raise ValueError(err_fmt.format("cond_model.factor"))
        res = read_image(
            semantic,
            max_wh,
            anchor=anchor,
            to_gray=True,
            resample=Image.NEAREST,
            normalize=False,
        )
        cond = torch.from_numpy(res.image).to(torch.long).to(self.device)
        cond = F.one_hot(cond, num_classes=in_channels)[0]
        cond = cond.half() if self.use_half else cond.float()
        cond = cond.permute(0, 3, 1, 2).contiguous()
        cond = self.get_cond(cond)
        size = self._get_identical_size_with(cond)
        return self.sample(
            1,
            export_path,
            size=size,
            original_size=res.original_size,
            alpha=res.alpha if alpha is None else alpha,
            cond=cond,
            num_steps=num_steps,
            clip_output=clip_output,
            verbose=verbose,
            **kwargs,
        )

    def load_context(self, *, ignore_lora: bool = True) -> ContextManager:
        class _:
            def __init__(self, api: DiffusionAPI):
                self.api = api
                self.m_ctrl = api.m.control_model
                self.m_cond = api.m.condition_model
                api.m.control_model = None
                api.m.condition_model = api.cond_model
                if not ignore_lora:
                    self.lora_checkpoints = None
                else:
                    if not isinstance(api.m, StableDiffusion):
                        msg = "currently only `StableDiffusion` supports `ignore_lora`"
                        raise ValueError(msg)
                    if not api.m.has_lora:
                        self.lora_checkpoints = None
                    else:
                        self.lora_checkpoints = api.m.get_lora_checkpoints()
                        api.m.cleanup_lora()

            def __enter__(self) -> DDPM:
                return self.api.m

            def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                if self.lora_checkpoints is not None:
                    assert isinstance(self.api.m, StableDiffusion)
                    self.api.m.restore_lora_from(self.lora_checkpoints)
                self.api.m.control_model = self.m_ctrl
                self.api.m.condition_model = self.m_cond

        return _(self)

    # lora

    def load_sd_lora(self, key: str, *, path: str) -> None:
        if not isinstance(self.m, StableDiffusion):
            raise ValueError("only `StableDiffusion` can use `load_sd_lora`")
        with self.load_context(ignore_lora=False):
            self.m.load_lora(key, path=path)

    def inject_sd_lora(self, *keys: str) -> None:
        if not isinstance(self.m, StableDiffusion):
            raise ValueError("only `StableDiffusion` can use `inject_sd_lora`")
        with self.load_context(ignore_lora=False):
            self.m.inject_lora(*keys)

    def cleanup_sd_lora(self) -> None:
        if not isinstance(self.m, StableDiffusion):
            raise ValueError("only `StableDiffusion` can use `cleanup_sd_lora`")
        with self.load_context(ignore_lora=False):
            self.m.cleanup_lora()

    def set_sd_lora_scales(self, scales: Dict[str, float]) -> None:
        if not isinstance(self.m, StableDiffusion):
            raise ValueError("only `StableDiffusion` can use `set_sd_lora_scales`")
        with self.load_context(ignore_lora=False):
            self.m.set_lora_scales(scales)

    # tomesd

    def set_tome_info(self, tome_info: Optional[Dict[str, Any]]) -> None:
        self.m.unet.set_tome_info(tome_info)

    # constructors

    @classmethod
    def from_sd(
        cls: Type[T],
        device: Optional[str] = None,
        *,
        use_amp: bool = False,
        use_half: bool = False,
    ) -> T:
        return cls.from_pipeline(ldm_sd(), device, use_amp=use_amp, use_half=use_half)

    @classmethod
    def from_sd_version(
        cls: Type[T],
        version: str,
        device: Optional[str] = None,
        *,
        use_amp: bool = False,
        use_half: bool = False,
        **kw: Any,
    ) -> T:
        m = ldm_sd_tag(get_sd_tag(version))
        return cls.from_pipeline(m, device, use_amp=use_amp, use_half=use_half, **kw)

    @classmethod
    def from_sd_anime(
        cls: Type[T],
        device: Optional[str] = None,
        *,
        use_amp: bool = False,
        use_half: bool = False,
    ) -> T:
        v = SDVersions.ANIME
        return cls.from_sd_version(v, device, use_amp=use_amp, use_half=use_half)

    @classmethod
    def from_sd_inpainting(
        cls: Type[T],
        device: Optional[str] = None,
        *,
        use_amp: bool = False,
        use_half: bool = False,
        **kw: Any,
    ) -> T:
        m = ldm_sd_inpainting()
        return cls.from_pipeline(m, device, use_amp=use_amp, use_half=use_half, **kw)

    @classmethod
    def from_sd_v2(
        cls: Type[T],
        device: Optional[str] = None,
        *,
        use_amp: bool = False,
        use_half: bool = False,
    ) -> T:
        return cls.from_pipeline(
            ldm_sd_v2(),
            device,
            use_amp=use_amp,
            use_half=use_half,
            clip_skip=1,
        )

    @classmethod
    def from_sd_v2_base(
        cls: Type[T],
        device: Optional[str] = None,
        *,
        use_amp: bool = False,
        use_half: bool = False,
    ) -> T:
        return cls.from_pipeline(
            ldm_sd_v2_base(),
            device,
            use_amp=use_amp,
            use_half=use_half,
            clip_skip=1,
        )

    @classmethod
    def from_celeba_hq(
        cls: Type[T],
        device: Optional[str] = None,
        *,
        use_amp: bool = False,
        use_half: bool = False,
    ) -> T:
        m = ldm_celeba_hq()
        return cls.from_pipeline(m, device, use_amp=use_amp, use_half=use_half)

    @classmethod
    def from_inpainting(
        cls: Type[T],
        device: Optional[str] = None,
        *,
        use_amp: bool = False,
        use_half: bool = False,
    ) -> T:
        m = ldm_inpainting()
        return cls.from_pipeline(m, device, use_amp=use_amp, use_half=use_half)

    @classmethod
    def from_sr(
        cls: Type[T],
        device: Optional[str] = None,
        *,
        use_amp: bool = False,
        use_half: bool = False,
    ) -> T:
        return cls.from_pipeline(ldm_sr(), device, use_amp=use_amp, use_half=use_half)

    @classmethod
    def from_semantic(
        cls: Type[T],
        device: Optional[str] = None,
        *,
        use_amp: bool = False,
        use_half: bool = False,
    ) -> T:
        m = ldm_semantic()
        return cls.from_pipeline(m, device, use_amp=use_amp, use_half=use_half)

    # internal

    def _get_z(self, img: arr_type) -> Tensor:
        img = 2.0 * img - 1.0
        z = img if isinstance(img, Tensor) else torch.from_numpy(img)
        if self.use_half:
            z = z.half()
        z = z.to(self.device)
        z = self.m._preprocess(z, deterministic=True)
        return z

    def _get_identical_size_with(self, pivot: Tensor) -> Tuple[int, int]:
        return tuple(  # type: ignore
            map(
                lambda n: n * self.size_info.factor,
                pivot.shape[-2:][::-1],
            )
        )

    def _set_seed_and_variations(
        self,
        seed: Optional[int],
        get_noise: Callable[[], Tensor],
        get_new_z: Callable[[Tensor], Tensor],
        variations: Optional[List[Tuple[int, float]]],
        variation_seed: Optional[int],
        variation_strength: Optional[float],
    ) -> Tuple[Tensor, Tensor]:
        if seed is None:
            seed = new_seed()
        seed = seed_everything(seed)
        self.latest_seed = seed
        z_noise = get_noise()
        self.latest_variation_seed = None
        if variations is not None:
            for v_seed, v_weight in variations:
                seed_everything(v_seed)
                v_noise = get_noise()
                z_noise = slerp(v_noise, z_noise, v_weight)
        if variation_strength is not None:
            random.seed()
            if variation_seed is None:
                variation_seed = new_seed()
            variation_seed = seed_everything(variation_seed)
            self.latest_variation_seed = variation_seed
            variation_noise = get_noise()
            z_noise = slerp(variation_noise, z_noise, variation_strength)
        z = get_new_z(z_noise)
        return z, z_noise

    def _update_clip_skip(self, clip_skip: int) -> None:
        if isinstance(self.cond_model, CLIPTextConditionModel):
            self.cond_model.clip_skip = clip_skip

    def _update_sampler_uncond(self, clip_skip: int) -> None:
        self._update_clip_skip(clip_skip)
        if self.cond_model is not None and self._original_raw_uncond is not None:
            cache = self._uncond_cache.get(clip_skip)
            if cache is not None:
                uncond = cache
            else:
                uncond = self.get_cond(self._original_raw_uncond)
                self._uncond_cache[clip_skip] = uncond
            self.m.sampler.unconditional_cond = uncond.to(self.device)

    def _get_masked_cond(
        self,
        image: Union[str, Image.Image],
        mask: Union[str, Image.Image],
        max_wh: int,
        anchor: int,
        mask_image_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        mask_cond_fn: Callable[[np.ndarray], Tensor],
        inpainting_settings: Optional[InpaintingSettings] = None,
    ) -> MaskedCond:
        # handle mask stuffs
        image_res = read_image(image, max_wh, anchor=anchor)
        mask_res = read_image(mask, max_wh, anchor=anchor, to_mask=True)
        cropped_res = get_cropped(image_res, mask_res, inpainting_settings)
        c_image = cropped_res.image
        c_mask = cropped_res.mask
        bool_mask = np.round(c_mask) >= 0.5
        remained_mask = (~bool_mask).astype(np.float16 if self.use_half else np.float32)
        remained_image = mask_image_fn(remained_mask, c_image)
        # construct condition tensor
        remained_cond = self._get_z(remained_image)
        latent_shape = remained_cond.shape[-2:]
        mask_cond = mask_cond_fn(bool_mask).to(torch.float32)
        mask_cond = F.interpolate(mask_cond, size=latent_shape)
        if self.use_half:
            mask_cond = mask_cond.half()
        mask_cond = mask_cond.to(self.device)
        return MaskedCond(
            c_image,
            c_mask,
            mask_cond,
            remained_cond,
            remained_image,
            remained_mask,
            image_res.alpha,
            image_res.original_size,
            image_res.original,
            mask_res.original,
            cropped_res.wh_ratio,
            cropped_res.crop_res,
        )

    def _q_sample(
        self,
        z: Tensor,
        num_steps: Optional[int],
        fidelity: float,
        seed: Optional[int],
        variations: Optional[List[Tuple[int, float]]] = None,
        variation_seed: Optional[int] = None,
        variation_strength: Optional[float] = None,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor, int]:
        if num_steps is None:
            num_steps = self.sampler.default_steps
        t = min(num_steps, round((1.0 - fidelity) * (num_steps + 1)))
        ts = get_timesteps(t, 1, z.device)
        if isinstance(self.sampler, (DDIMMixin, KSamplerMixin, DPMSolver)):
            kw = shallow_copy_dict(self.sampler.sample_kwargs)
            kw["total_step"] = num_steps
            safe_execute(self.sampler._reset_buffers, kw)
        z, noise = self._set_seed_and_variations(
            seed,
            lambda: torch.randn_like(z),
            lambda noise_: self.sampler.q_sample(z, ts, noise_),
            variations,
            variation_seed,
            variation_strength,
        )
        start_step = num_steps - t
        return z, noise, start_step

    def _img2img(
        self,
        z: Tensor,
        export_path: Optional[str] = None,
        *,
        z_ref: Optional[Tensor] = None,
        z_ref_mask: Optional[Tensor] = None,
        original_size: Optional[Tuple[int, int]] = None,
        alpha: Optional[np.ndarray] = None,
        cond: Optional[Any] = None,
        num_steps: Optional[int] = None,
        clip_output: bool = True,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        with switch_sampler_context(self, kwargs.get("sampler")):
            z, noise, start_step = self._q_sample(z, num_steps, **kwargs)
            kwargs["start_step"] = start_step
            return self.sample(
                z.shape[0],
                export_path,
                z=z,
                z_ref=z_ref,
                z_ref_mask=z_ref_mask,
                z_ref_noise=None if z_ref is None else noise,
                original_size=original_size,
                alpha=alpha,
                cond=cond,
                num_steps=num_steps,
                clip_output=clip_output,
                verbose=verbose,
                **kwargs,
            )

    def _get_highres_latent(self, z: Tensor, highres_info: Dict[str, Any]) -> Tensor:
        upscale_factor = highres_info["upscale_factor"]
        shrink_factor = self.size_info.factor
        max_wh = round(highres_info["max_wh"] / shrink_factor)
        h, w = z.shape[-2:]
        upscaled = round(w * upscale_factor), round(h * upscale_factor)
        w, h = get_size(upscaled, 64 // shrink_factor, max_wh)  # type: ignore
        return F.interpolate(z, size=(h, w), mode="bilinear", antialias=False)