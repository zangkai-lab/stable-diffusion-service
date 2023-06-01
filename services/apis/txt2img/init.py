import os

from functools import partial

from tools.enum.apis import SDVersions


def init_sd(init_to_cpu: bool) -> ControlledDiffusionAPI:
    version = SDVersions.v1_5
    init_fn = partial(ControlledDiffusionAPI.from_sd_version, version, lazy=True)
    m: ControlledDiffusionAPI = _get(init_fn, init_to_cpu)
    m.sd_weights.limit = -1
    m.current_sd_version = version
    print("> registering base sd")
    m.prepare_sd([version])
    m.sd_weights.register(BaseSDTag, _base_sd_path())

    print("> prepare ControlNet weights")
    m.prepare_control_defaults()
    print("> prepare ControlNet Annotators")
    m.prepare_annotators()
    print("> warmup ControlNet")
    m.switch_control(*m.available_control_hints)

    return m
