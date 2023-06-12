import os
from typing import Any, Dict, Optional, Tuple

from collections import OrderedDict

from models.pipeline.pipeline_base import Block
from models.config.train_config import DLConfig
from models.data.data import IData

from tools.utils.tool import truncate_string_to_length


@Block.register("report")
class ReportBlock(Block):
    config: DLConfig
    report_file = "report.txt"

    def build(self, config: DLConfig) -> None:
        self.config = config

    def run(self, data: IData, _defaults: OrderedDict, **kwargs: Any) -> None:
        if not self.is_local_rank_0 or self.training_workspace is None:
            return
        self._report_messages(
            "Internal Default Configurations Used by `carefree-learn`",
            _defaults,
            self.training_workspace,
        )
        original = self.config.__class__().asdict()
        external_configs: Dict[str, Any] = {}
        for k, v in self.config.asdict().items():
            if k in _defaults:
                continue
            ov = original[k]
            if v != ov:
                external_configs[k] = v
        self._report_messages(
            "External Configurations",
            external_configs,
            self.training_workspace,
        )

    def _report_messages(
        self,
        title: str,
        messages: Dict[str, Any],
        report_folder: str,
    ) -> None:
        def _stringify_item(
            item: Tuple[str, Any],
            prefix: Optional[str] = None,
            depth: int = 0,
        ) -> str:
            key, value = item
            if prefix is not None:
                key = f"{prefix}{key}"
            if not isinstance(value, dict) or not value or depth >= 2:
                key = truncate_string_to_length(key, span)
                return f"{key:>{span}s}   |   {value}"
            prefix = f"{key}."
            items = [
                _stringify_item((vk, vv), prefix, depth=depth + 1)
                for vk, vv in value.items()
            ]
            return "\n".join(items)

        span = 64
        length = 2 * span
        msg = "\n".join(
            [
                "=" * length,
                f"{title:^{length}s}",
                "-" * length,
                "\n".join(map(_stringify_item, messages.items())),
                "-" * length,
            ]
        )
        print(msg)
        if report_folder is not None:
            with open(os.path.join(report_folder, self.report_file), "a") as f:
                f.write(msg + "\n")