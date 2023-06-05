import torch
import onnx
import inspect
import torch.nn as nn

from torch import Tensor
from abc import ABCMeta
from typing import Any, Optional, Tuple, Union, List, Dict, Callable, Type
from onnxsim import simplify as onnx_simplify

from models.model.constant import INPUT_KEY, PREDICTIONS_KEY
from models.model.mode_context import eval_context

from tools.bases.register import WithRegister
from tools.utils.type import tensor_dict_type, forward_results_type
from tools.utils.device import get_device
from tools.utils.icopy import shallow_copy_dict
from tools.utils.fix import fix_denormal_states


model_dict: Dict[str, Type["IDLModel"]] = {}


def get_inputs(model: onnx.ModelProto) -> List[onnx.ValueInfoProto]:
    initializer_names = [x.name for x in model.graph.initializer]
    return [inp for inp in model.graph.input if inp.name not in initializer_names]


def get_input_names(model: onnx.ModelProto) -> List[str]:
    input_names = [inp.name for inp in get_inputs(model)]
    return input_names


def get_requirements(fn: Any) -> List[str]:
    if isinstance(fn, type):
        fn = fn.__init__  # type: ignore
    requirements = []
    signature = inspect.signature(fn)
    for k, param in signature.parameters.items():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        if param.default is not inspect.Parameter.empty:
            continue
        requirements.append(k)
    return requirements


class IWithRequirements:
    @classmethod
    def requirements(cls) -> List[str]:
        requirements = get_requirements(cls)
        requirements.remove("self")
        return requirements


class IDLModel(
    nn.Module,
    WithRegister["IDLModel"],
    IWithRequirements,
    metaclass=ABCMeta,
):
    d = model_dict

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()

    # optional callbacks

    def init_with_trainer(self, trainer: "ITrainer") -> None:
        pass

    def permute_trainer_config(self, trainer_config: "TrainerConfig") -> None:
        pass

    def get_forward_args(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional["TrainerState"] = None,
        **kwargs: Any,
    ) -> Tuple[Any, ...]:
        return (batch[INPUT_KEY],)

    def postprocess(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        forward_results: forward_results_type,
        state: Optional["TrainerState"] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        if isinstance(forward_results, dict):
            return forward_results
        if isinstance(forward_results, Tensor):
            return {PREDICTIONS_KEY: forward_results}
        raise ValueError(f"unrecognized forward results occurred: {forward_results}")

    # api

    def run(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional["TrainerState"] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        args = self.get_forward_args(batch_idx, batch, state, **kwargs)
        forward_results = self(*args)
        outputs = self.postprocess(batch_idx, batch, forward_results, state, **kwargs)
        return outputs

    def onnx_forward(self, batch: tensor_dict_type) -> Any:
        return self.run(0, batch)

    def summary_forward(self, batch: tensor_dict_type) -> None:
        self.onnx_forward(batch)

    def to_onnx(
        self,
        export_file: str,
        input_sample: tensor_dict_type,
        dynamic_axes: Optional[Union[List[int], Dict[int, str]]] = None,
        *,
        opset: int = 11,
        simplify: bool = True,
        forward_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        output_names: Optional[List[str]] = None,
        num_samples: Optional[int] = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> "IDLModel":
        # prepare
        device = get_device(self)
        model = self.cpu()
        if num_samples is not None:
            input_sample = {k: v[:num_samples] for k, v in input_sample.items()}
        onnx_forward = forward_fn or model.onnx_forward
        input_names = sorted(input_sample.keys())
        if output_names is None:
            if forward_fn is not None:
                msg = "`output_names` should be provided when `forward_fn` is provided"
                raise ValueError(msg)
            with eval_context(model):
                forward_results = onnx_forward(shallow_copy_dict(input_sample))
            if not isinstance(forward_results, dict):
                forward_results = {PREDICTIONS_KEY: forward_results}
            output_names = sorted(forward_results.keys())
        # setup
        kwargs = shallow_copy_dict(kwargs)
        kwargs["input_names"] = input_names
        kwargs["output_names"] = output_names
        kwargs["opset_version"] = opset
        kwargs["export_params"] = True
        kwargs["do_constant_folding"] = True
        if dynamic_axes is None:
            dynamic_axes = {}
        elif isinstance(dynamic_axes, list):
            dynamic_axes = {axis: f"axis.{axis}" for axis in dynamic_axes}
        if num_samples is None:
            dynamic_axes[0] = "batch_size"
        dynamic_axes_settings = {}
        for name in input_names + output_names:
            dynamic_axes_settings[name] = dynamic_axes
        kwargs["dynamic_axes"] = dynamic_axes_settings
        kwargs["verbose"] = verbose
        # export

        class ONNXWrapper(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = model

            def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
                rs = onnx_forward(batch)
                if isinstance(rs, Tensor):
                    return {k: rs for k in output_names}  # type: ignore
                return {k: rs[k] for k in output_names}  # type: ignore

        m_onnx = ONNXWrapper()
        original_states = model.state_dict()
        fixed_states = fix_denormal_states(original_states, verbose=verbose)
        with eval_context(m_onnx):
            model.load_state_dict(fixed_states)
            torch.onnx.export(
                m_onnx,
                ({k: input_sample[k] for k in input_names}, {}),
                export_file,
                **shallow_copy_dict(kwargs),
            )
            model.load_state_dict(original_states)
            if not simplify:
                return self.to(device)
            if onnx is None:
                print(
                    "`onnx` is not installed, "
                    "so the exported onnx model will not be simplified"
                )
                return self.to(device)
            if onnx_simplify is None or get_input_names is None:
                print(
                    "`onnx-simplifier` is not installed, "
                    "so the exported onnx model will not be simplified"
                )
                return self.to(device)
            try:
                onnx_model = onnx.load(export_file)
                final_input_names = get_input_names(onnx_model)
                model_simplified, check = onnx_simplify(
                    onnx_model,
                    test_input_shapes={
                        name: tensor.shape
                        for name, tensor in input_sample.items()
                        if name in final_input_names
                    },
                )
            except Exception as err:
                if verbose:
                    print(f"Failed to simplify ONNX model ({err})")
                model_simplified = None
                check = False
            if verbose:
                tag = " " if check else " not "
                print(f"Simplified ONNX model is{tag}validated!")
            if check and model_simplified is not None:
                onnx.save(model_simplified, export_file)
        return self.to(device)