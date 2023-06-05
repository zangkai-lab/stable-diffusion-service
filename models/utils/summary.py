import torch
import torch.nn as nn

from torch import Tensor
from collections import OrderedDict
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Set

from tools.utils.type import tensor_dict_type
from models.model.mode_context import eval_context
from models.model.constant import INPUT_KEY
from tools.utils.tool import prod, truncate_string_to_length


def summary(
    model: nn.Module,
    sample_batch: tensor_dict_type,
    *,
    return_only: bool = False,
) -> str:
    def _get_param_counts(module_: nn.Module) -> Tuple[int, int]:
        num_params = 0
        num_trainable_params = 0
        for p in module_.parameters():
            local_num_params = int(round(prod(p.data.shape)))
            num_params += local_num_params
            if p.requires_grad:
                num_trainable_params += local_num_params
        return num_params, num_trainable_params

    def register_hook(module: nn.Module) -> None:
        def inject_output_shape(output: Any, res: Dict[int, Any]) -> None:
            idx = 0 if not res else max(res)
            if isinstance(output, Tensor):
                o_shape = list(output.shape)
                if o_shape:
                    o_shape[0] = -1
                res[idx + 1] = o_shape
                return
            if isinstance(output, (list, tuple)):
                o_res = res[idx + 1] = {}
                for o in output:
                    inject_output_shape(o, o_res)

        def hook(module_: nn.Module, inp: Any, output: Any) -> None:
            m_name = module_names.get(module_)
            if m_name is None:
                return

            if not inp:
                return
            inp = inp[0]
            if not isinstance(inp, Tensor):
                return

            m_dict: OrderedDict[str, Any] = OrderedDict()
            m_dict["input_shape"] = list(inp.shape)
            if len(m_dict["input_shape"]) > 0:
                m_dict["input_shape"][0] = -1
            output_shape_res = m_dict["output_shape"] = {}
            inject_output_shape(output, output_shape_res)

            num_params_, num_trainable_params_ = _get_param_counts(module_)
            m_dict["num_params"] = num_params_
            m_dict["num_trainable_params"] = num_trainable_params_
            raw_summary_dict[m_name] = m_dict

        if not isinstance(module, torch.jit.ScriptModule):
            hooks.append(module.register_forward_hook(hook))

    # get names
    def _inject_names(m: nn.Module, previous_names: List[str]) -> None:
        info_list = []
        for child in m.children():
            current_names = previous_names + [type(child).__name__]
            current_name = ".".join(current_names)
            module_names[child] = current_name
            info_list.append((child, current_name, current_names))
        counts: Dict[str, int] = defaultdict(int)
        idx_mapping: Dict[nn.Module, int] = {}
        for child, current_name, _ in info_list:
            idx_mapping[child] = counts[current_name]
            counts[current_name] += 1
        for child, current_name, current_names in info_list:
            if counts[current_name] == 1:
                continue
            current_name = f"{current_name}-{idx_mapping[child]}"
            module_names[child] = current_name
            current_names[-1] = current_name.split(".")[-1]
        for child, _, current_names in info_list:
            _inject_names(child, current_names)

    module_names: OrderedDict[nn.Module, str] = OrderedDict()
    existing_names: Set[str] = set()

    def _get_name(original: str) -> str:
        count = 0
        final_name = original
        while final_name in existing_names:
            count += 1
            final_name = f"{original}_{count}"
        existing_names.add(final_name)
        return final_name

    model_name = _get_name(type(model).__name__)
    module_names[model] = model_name
    _inject_names(model, [model_name])

    # create properties
    raw_summary_dict: OrderedDict[str, Any] = OrderedDict()
    hooks: List[Any] = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    with eval_context(model, use_grad=None):
        if not hasattr(model, "summary_forward"):
            model.run(0, sample_batch)
        else:
            model.summary_forward(sample_batch)  # type: ignore
        for param in model.parameters():
            param.grad = None

    # remove these hooks
    for h in hooks:
        h.remove()

    # get hierarchy
    hierarchy: OrderedDict[str, Any] = OrderedDict()
    for key in raw_summary_dict:
        split = key.split(".")
        d = hierarchy
        for elem in split[:-1]:
            d = d.setdefault(elem, OrderedDict())
        d.setdefault(split[-1], None)

    # reconstruct summary_dict
    def _inject_summary(current_hierarchy: Any, previous_keys: List[str]) -> None:
        if previous_keys and not previous_keys[-1]:
            previous_keys.pop()
        current_layer = len(previous_keys)
        current_count = hierarchy_counts.get(current_layer, 0)
        prefix = "  " * current_layer
        for k, v in current_hierarchy.items():
            current_keys = previous_keys + [k]
            concat_k = ".".join(current_keys)
            current_summary = raw_summary_dict.get(concat_k)
            summary_dict[f"{prefix}{k}-{current_count}"] = current_summary
            hierarchy_counts[current_layer] = current_count + 1
            if v is not None:
                _inject_summary(v, current_keys)

    hierarchy_counts: Dict[int, int] = {}
    summary_dict: OrderedDict[str, Any] = OrderedDict()
    _inject_summary(hierarchy, [])

    line_length = 120
    messages = ["=" * line_length]
    line_format = "{:30}  {:>20} {:>40} {:>20}"
    headers = "Layer (type)", "Input Shape", "Output Shape", "Trainable Param #"
    messages.append(line_format.format(*headers))
    messages.append("-" * line_length)
    total_output = 0
    for layer, layer_summary in summary_dict.items():
        layer_name = "-".join(layer.split("-")[:-1])
        valid_layer_name = layer_name.strip()
        num_spaces = len(layer_name) - len(valid_layer_name)
        valid_layer_name = truncate_string_to_length(valid_layer_name, 30 - num_spaces)
        layer_name = " " * num_spaces + valid_layer_name
        if layer_summary is None:
            messages.append(line_format.format(layer_name, "", "", ""))
        else:
            is_title = True
            all_output_shapes: List[List[int]] = []

            def _inject(output_shape_item: Dict[int, Any], prefix: str) -> None:
                only_one = len(output_shape_item) == 1
                for i, idx in enumerate(sorted(output_shape_item)):
                    if not prefix and only_one:
                        idx_prefix = ""
                    else:
                        idx_prefix = f"{prefix}{idx}."
                    value = output_shape_item[idx]
                    if isinstance(value, dict):
                        _inject(value, idx_prefix)
                        continue
                    output_shape_str = f"{idx_prefix} {str(value):>16s}"
                    ntp_str = "{0:,}".format(layer_summary["num_trainable_params"])
                    nonlocal is_title
                    messages.append(
                        line_format.format(
                            layer_name if is_title else "",
                            str(layer_summary["input_shape"]) if is_title else "",
                            output_shape_str,
                            ntp_str if is_title else "",
                        )
                    )
                    is_title = False
                    all_output_shapes.append(value)

            _inject(layer_summary["output_shape"], "")
            for shape in all_output_shapes:
                total_output += prod(shape)

    total_params, trainable_params = _get_param_counts(model)
    # assume 4 bytes/number (float on cuda).
    x_batch = sample_batch[INPUT_KEY]
    get_size = lambda t: abs(prod(t.shape[1:]) * 4.0 / (1024**2.0))
    if not isinstance(x_batch, list):
        x_batch = [x_batch]
    total_input_size = sum(map(get_size, x_batch))
    # x2 for gradients
    total_output_size = abs(2.0 * total_output * 4.0 / (1024**2.0))
    total_params_size = abs(total_params * 4.0 / (1024**2.0))
    total_size = total_params_size + total_output_size + total_input_size

    non_trainable_params = total_params - trainable_params
    messages.append("=" * line_length)
    messages.append("Total params: {0:,}".format(total_params))
    messages.append("Trainable params: {0:,}".format(trainable_params))
    messages.append("Non-trainable params: {0:,}".format(non_trainable_params))
    messages.append("-" * line_length)
    messages.append("Input size (MB): %0.2f" % total_input_size)
    messages.append("Forward/backward pass size (MB): %0.2f" % total_output_size)
    messages.append("Params size (MB): %0.2f" % total_params_size)
    messages.append("Estimated Total Size (MB): %0.2f" % total_size)
    messages.append("-" * line_length)
    msg = "\n".join(messages)
    if not return_only:
        print(msg)
    return msg