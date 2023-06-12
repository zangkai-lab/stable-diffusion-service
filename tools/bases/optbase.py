import os
import json

from abc import ABC, abstractmethod
from typing import Dict, Any

from tools.utils.icopy import shallow_copy_dict


class OPTBase(ABC):
    def __init__(self):
        self._opt = self.defaults
        self.update_from_env()

    @property
    @abstractmethod
    def env_key(self) -> str:
        pass

    @property
    @abstractmethod
    def defaults(self) -> Dict[str, Any]:
        pass

    def __getattr__(self, __name: str) -> Any:
        return self._opt[__name]

    def update_from_env(self) -> None:
        env_opt_json = os.environ.get(self.env_key)
        if env_opt_json is not None:
            self._opt.update(json.loads(env_opt_json))

    def opt_context(self, increment: Dict[str, Any]) -> Any:
        class _:
            def __init__(self) -> None:
                self._increment = increment
                self._backup = shallow_copy_dict(instance._opt)

            def __enter__(self) -> None:
                instance._opt.update(self._increment)

            def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                instance._opt.update(self._backup)

        instance = self
        return _()

    def opt_env_context(self, increment: Dict[str, Any]) -> Any:
        class _:
            def __init__(self) -> None:
                self._increment = increment
                self._backup = os.environ.get(instance.env_key)

            def __enter__(self) -> None:
                os.environ[instance.env_key] = json.dumps(self._increment)

            def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                if self._backup is None:
                    del os.environ[instance.env_key]
                else:
                    os.environ[instance.env_key] = self._backup

        instance = self
        return _()