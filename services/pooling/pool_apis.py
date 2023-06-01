import torch

from typing import Any, Dict, Optional, Protocol

from tools.bases.pooling import ILoadablePool, ILoadableItem
from tools.enum.APIS import APIs


class IAPI:
    def to(self, device: str, *, use_half: bool) -> None:
        pass


class APIInit(Protocol):
    def __call__(self, init_to_cpu: bool) -> IAPI:
        pass


class LoadableAPI(ILoadableItem[IAPI]):
    def __init__(
        self,
        init_fn: APIInit,
        *,
        init: bool = False,
        force_not_lazy: bool = False,
        has_annotator: bool = False,
    ):
        super().__init__(lambda: init_fn(self.init_to_cpu), init=init)
        self.force_not_lazy = force_not_lazy
        self.has_annotator = has_annotator

    @property
    def lazy(self) -> bool:
        return False and not self.force_not_lazy

    @property
    def init_to_cpu(self) -> bool:
        return self.lazy or False

    @property
    def need_change_device(self) -> bool:
        return self.lazy and not False

    @property
    def annotator_kwargs(self) -> Dict[str, Any]:
        return {"no_annotator": True} if self.has_annotator else {}

    def load(self, *, no_change: bool = False, **kwargs: Any) -> IAPI:
        super().load()
        if not no_change and self.need_change_device:
            self._item.to("cuda:0", use_half=True)
        return self._item

    def cleanup(self) -> None:
        if self.need_change_device:
            self._item.to("cpu", use_half=False)
            torch.cuda.empty_cache()

    def unload(self) -> None:
        self.cleanup()
        return super().unload()


class APIPool(ILoadablePool[IAPI]):
    def register(self, key: str, init_fn: APIInit) -> None:
        def _init(init: bool) -> LoadableAPI:
            # APIs.SD是需要强制加载的
            kw = dict(
                force_not_lazy=key in APIs.SD,
                has_annotator=key in APIs.SD,
            )
            api = LoadableAPI(init_fn, init=False, **kw)
            if init:
                print("> init", key, "(lazy)" if api.lazy else "")
                api.load(no_change=api.lazy)
            return api

        if key in self:
            return
        return super().register(key, _init)

    def cleanup(self, key: str) -> None:
        loadable_api: Optional[LoadableAPI] = self.pool.get(key)
        if loadable_api is None:
            raise ValueError(f"key '{key}' does not exist")
        loadable_api.cleanup()


api_pool = APIPool()
