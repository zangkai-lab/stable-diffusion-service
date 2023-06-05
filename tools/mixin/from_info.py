from typing import Any, Dict


class PureFromInfoMixin:
    def from_info(self, info: Dict[str, Any]) -> None:
        for k, v in info.items():
            setattr(self, k, v)