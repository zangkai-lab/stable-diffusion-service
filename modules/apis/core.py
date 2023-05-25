from enum import Enum


class APIs(str, Enum):
    SD = "sd"


class APIPool(ILoadablePool[IAPI]):
    def register(self, key: str, init_fn: APIInit) -> None:
        def _init(init: bool) -> LoadableAPI:
            kw = dict(
                force_not_lazy=key in (APIs.SD, APIs.SD_INPAINTING),
                has_annotator=key in (APIs.SD, APIs.SD_INPAINTING),
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

    def need_change_device(self, key: str) -> bool:
        loadable_api: Optional[LoadableAPI] = self.pool.get(key)
        if loadable_api is None:
            raise ValueError(f"key '{key}' does not exist")
        return loadable_api.need_change_device

    def update_limit(self) -> None:
        self.limit = pool_limit()
