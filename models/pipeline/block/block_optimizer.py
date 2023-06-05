import os
import torch
import torch.nn as nn

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Any, Dict, List, Optional, Set, Type, NamedTuple

from models.pipeline.pipeline_base import Block
from models.config.train_config import DLConfig
from models.scheduler.scheduler_warm import WarmupScheduler, scheduler_requires_metric, scheduler_dict
from models.model.optimizer.optimizer_core import optimizer_dict
from models.model.custom import ModelWithCustomSteps
from models.pipeline.block.block_stateinfo import ExtractStateInfoBlock
from models.pipeline.block.block_model import BuildModelBlock
from models.pipeline.block.block_stateinfo import StateInfo

from tools.utils.update import update_dict


class OptimizerPack(NamedTuple):
    scope: str
    optimizer_name: str
    scheduler_name: Optional[str] = None
    optimizer_config: Optional[Dict[str, Any]] = None
    scheduler_config: Optional[Dict[str, Any]] = None


class DefaultOptimizerSettings(NamedTuple):
    lr: float = 1.0e-3
    optimizer_name: str = "adam"
    scheduler_name: Optional[str] = "warmup"
    optimizer_config: Optional[Dict[str, Any]] = None
    scheduler_config: Optional[Dict[str, Any]] = None

    def get_opt_pack(self, state_info: Optional[StateInfo]) -> OptimizerPack:
        optimizer_config = self.optimizer_config or {}
        scheduler_config = self.scheduler_config or {}
        if self.scheduler_name != "warmup":
            optimizer_config.setdefault("lr", self.lr)
        else:
            multiplier = scheduler_config.setdefault("multiplier", 3)
            optimizer_config.setdefault("lr", self.lr / multiplier)
            if state_info is None:
                scheduler_config.setdefault("warmup_step", 1000)
            else:
                default_max_warmup_step = int(round(3.0e5 / state_info.batch_size))
                scheduler_config.setdefault(
                    "warmup_step",
                    min(default_max_warmup_step, 10 * state_info.num_batches),
                )
        if self.optimizer_name == "nag":
            optimizer_config.setdefault("momentum", 0.999)
            optimizer_config.setdefault("weight_decay", 1e-7)
        return OptimizerPack(
            "all",
            self.optimizer_name,
            self.scheduler_name,
            optimizer_config,
            scheduler_config,
        )

    def update_opt_pack(
        self,
        state_info: Optional[StateInfo],
        pack: OptimizerPack,
    ) -> OptimizerPack:
        self_pack = self.get_opt_pack(state_info)
        opt_config = pack.optimizer_config or {}
        sch_config = pack.scheduler_config or {}
        if self_pack.optimizer_name != pack.optimizer_name:
            opt_config.setdefault("lr", self.lr)
        else:
            opt_config = update_dict(opt_config, self_pack.optimizer_config)
        if self_pack.scheduler_name == pack.scheduler_name:
            sch_config = update_dict(sch_config, self_pack.scheduler_config)
        return OptimizerPack(
            pack.scope,
            pack.optimizer_name,
            pack.scheduler_name,
            opt_config,
            sch_config,
        )


@Block.register("build_optimizers")
class BuildOptimizersBlock(Block):
    config: DLConfig
    optimizers: Dict[str, Optimizer]
    schedulers: Dict[str, Optional[_LRScheduler]]
    schedulers_requires_metric: Set[str]

    def build(self, config: DLConfig) -> None:
        self.config = config
        state_info = self.extract_state_info.state_info
        # default settings
        settings: Dict[str, Any] = {}
        if config.lr is not None:
            settings["lr"] = config.lr
        if config.optimizer_name is not None:
            settings["optimizer_name"] = config.optimizer_name
        if config.scheduler_name is not None:
            if config.scheduler_name == "none":
                config.scheduler_name = None
            settings["scheduler_name"] = config.scheduler_name
        if config.optimizer_config is not None:
            settings["optimizer_config"] = config.optimizer_config
        if config.scheduler_config is not None:
            settings["scheduler_config"] = config.scheduler_config
        default_opt_settings = DefaultOptimizerSettings(**settings)
        # build
        optimizer_settings = config.optimizer_settings
        if optimizer_settings is None:
            optimizer_packs = [default_opt_settings.get_opt_pack(state_info)]
        else:
            optimizer_packs = []
            for key, settings in optimizer_settings.items():
                optimizer = settings.get("optimizer")
                if optimizer is None:
                    raise ValueError(f"optimizer must be provided (key={key})")
                optimizer_packs.append(
                    OptimizerPack(
                        key,
                        optimizer,
                        settings.get("scheduler"),
                        settings.get("optimizer_config"),
                        settings.get("scheduler_config"),
                    )
                )
        # initialize
        self.optimizers = {}
        self.schedulers = {}
        for pack in optimizer_packs:
            pack = default_opt_settings.update_opt_pack(state_info, pack)
            opt = self._define_optimizer(pack)
            self._define_scheduler(opt, pack)
        # check requires metric
        self.schedulers_requires_metric = set()
        for key, scheduler in self.schedulers.items():
            if scheduler is None:
                continue
            if isinstance(scheduler, WarmupScheduler):
                scheduler = scheduler.scheduler_afterwards
            if scheduler is not None and scheduler_requires_metric(scheduler):
                self.schedulers_requires_metric.add(key)

    @property
    def requirements(self) -> List[Type[Block]]:
        return [ExtractStateInfoBlock, BuildModelBlock]

    @property
    def build_model(self) -> BuildModelBlock:
        return self.get_previous(BuildModelBlock)

    @property
    def extract_state_info(self) -> ExtractStateInfoBlock:
        return self.get_previous(ExtractStateInfoBlock)

    def default_lr_configs(
        self,
        optimizer: Optimizer,
        optimizer_config: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        state_info = self.extract_state_info.state_info
        opt_lr = optimizer_config["lr"]
        # step
        step_default_cfg = {"step_size": 10 * state_info.num_batches}
        # exponential
        exp_gamma = (0.1**0.1) ** (1.0 / state_info.num_batches)
        exp_default_cfg = {"gamma": exp_gamma}
        # cyclic
        cyclic_default_cfg = {
            "base_lr": opt_lr,
            "max_lr": 1.0e-8,
            "step_size_up": 10 * state_info.num_batches,
            "gamma": exp_gamma,
        }
        if "momentum" not in optimizer.defaults:
            cyclic_default_cfg["cycle_momentum"] = False
        # cosine
        cosine_default_cfg = {
            "eta_min": 1.0e-8,
            "T_max": 10 * state_info.num_batches,
        }
        # cosine restarts
        cosine_restarts_default_cfg = {
            "eta_min": 1.0e-8,
            "T_0": 10 * state_info.num_batches,
        }
        # plateau
        plateau_default_cfg = {
            "mode": "max",
            "min_lr": 1.0e-8,
            "verbose": False,
            "patience": max(
                10 * state_info.num_step_per_snapshot,
                state_info.snapshot_start_step,
            ),
        }
        return {
            "step": step_default_cfg,
            "exponential": exp_default_cfg,
            "cyclic": cyclic_default_cfg,
            "cosine": cosine_default_cfg,
            "cosine_restarts": cosine_restarts_default_cfg,
            "plateau": plateau_default_cfg,
        }

    def _define_optimizer(self, pack: OptimizerPack) -> Optimizer:
        model = self.build_model.model
        if pack.scope == "all":
            if isinstance(model, ModelWithCustomSteps) and model.custom_params_groups:
                if self.config.use_zero and self.is_local_rank_0:
                    print(
                        "currently PyTorch does not support "
                        "using ZeRO with parameter groups, so ZeRO will be disabled"
                    )
                    self.config.use_zero = False
                parameters = model.params_groups(model)
            else:
                parameters = [p for p in model.parameters() if p.requires_grad]
        else:
            attr = model
            scopes = pack.scope.split(".")
            for scope in scopes:
                new_attr = getattr(attr, scope, None)
                if new_attr is None:
                    raise ValueError(f"'{attr}' has no scope '{scope}'")
                attr = new_attr
            if not isinstance(attr, nn.Module):
                parameters = attr
            else:
                parameters = attr.parameters()
        optimizer_base = optimizer_dict[pack.optimizer_name]
        opt_config = pack.optimizer_config or {}
        opt = optimizer_base(parameters, **opt_config)
        self.optimizers[pack.scope] = opt
        return opt

    def _define_scheduler(self, optimizer: Optimizer, pack: OptimizerPack) -> None:
        if pack.scheduler_name is None:
            self.schedulers[pack.scope] = None
        else:
            scheduler = pack.scheduler_name
            opt_config = pack.optimizer_config or {}
            scheduler_config = pack.scheduler_config or {}
            default_lr_configs = self.default_lr_configs(optimizer, opt_config)
            default_lr_config = default_lr_configs.get(scheduler)
            if default_lr_config is not None:
                scheduler_config = update_dict(scheduler_config, default_lr_config)
            if scheduler == "warmup":
                sab = scheduler_config.get("scheduler_afterwards_base", "plateau")
                if sab == "warmup":
                    raise ValueError("warmup should not be used inside a warmup")
                sac = scheduler_config.get("scheduler_afterwards_config", {})
                default_lr_config = default_lr_configs.get(sab)
                sac = update_dict(sac, default_lr_config or {})
                sab = scheduler_dict[sab]
                scheduler_config["scheduler_afterwards_base"] = sab
                scheduler_config["scheduler_afterwards_config"] = sac
            scheduler_base = scheduler_dict[scheduler]
            self.schedulers[pack.scope] = scheduler_base(optimizer, **scheduler_config)


@Block.register("serialize_optimizer")
class SerializeOptimizerBlock(Block):
    optimizer_file = "optimizers.pt"
    scheduler_file = "schedulers.pt"

    def build(self, config: DLConfig) -> None:
        pass

    @property
    def requirements(self) -> List[Type[Block]]:
        return [BuildOptimizersBlock]

    @property
    def build_optimizers(self) -> BuildOptimizersBlock:
        return self.get_previous(BuildOptimizersBlock)

    def save_extra(self, folder: str) -> None:
        optims = self.build_optimizers.optimizers
        scheds = self.build_optimizers.schedulers
        opt_d = {k: v.state_dict() for k, v in optims.items()}
        sch_d = {k: None if v is None else v.state_dict() for k, v in scheds.items()}
        os.makedirs(folder, exist_ok=True)
        torch.save(opt_d, os.path.join(folder, self.optimizer_file))
        torch.save(sch_d, os.path.join(folder, self.scheduler_file))

    def load_from(self, folder: str) -> None:
        optimizers = self.build_optimizers.optimizers
        schedulers = self.build_optimizers.schedulers
        opt_d = torch.load(os.path.join(folder, self.optimizer_file))
        sch_d = torch.load(os.path.join(folder, self.scheduler_file))
        for k, states in opt_d.items():
            optimizers[k].load_state_dict(states)
        for k, states in sch_d.items():
            k_sch = schedulers[k]
            if k_sch is not None:
                k_sch.load_state_dict(states)