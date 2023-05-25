import gc
import time

from typing import Any
from typing import Dict
from typing import Generic
from typing import TypeVar
from typing import Callable
from typing import Optional

from .utils import sort_dict_by_value

TIME_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"
TItem = TypeVar("TItem")


class ILoadableItem(Generic[TItem]):
    _item: Optional[TItem]

    def __init__(self, init_fn: Callable[[], TItem], *, init: bool = False):
        self.init_fn = init_fn
        self.load_time = time.time()
        self._item = init_fn() if init else None

    def load(self, **kwargs: Any) -> TItem:
        self.load_time = time.time()
        if self._item is None:
            self._item = self.init_fn()
        return self._item

    def unload(self) -> None:
        self._item = None
        gc.collect()


class ILoadablePool(Generic[TItem]):
    pool: Dict[str, ILoadableItem]
    activated: Dict[str, ILoadableItem]

    # set `limit` to negative values to indicate 'no limit'
    def __init__(self, limit: int = -1):
        self.pool = {}
        self.activated = {}
        self.limit = limit
        if limit == 0:
            raise ValueError(
                "limit should either be negative "
                "(which indicates 'no limit') or be positive"
            )

    # __contains__是一个魔术方法。当使用in操作符检查一个元素是否在一个对象中时，Python解释器会调用__contains__方法
    def __contains__(self, key: str) -> bool:
        return key in self.pool

    # 在满足注册数量限制的情况下注册新的key
    def register(self, key: str, init_fn: Callable[[bool], ILoadableItem]) -> None:
        if key in self.pool:
            raise ValueError(f"key '{key}' already exists")
        init = self.limit < 0 or len(self.activated) < self.limit
        loadable_item = init_fn(init)
        self.pool[key] = loadable_item
        if init:
            self.activated[key] = loadable_item

    # 在满足注册数量限制的情况下注册新的key
    def get(self, key: str, **kwargs: Any) -> TItem:
        loadable_item = self.pool.get(key)
        if loadable_item is None:
            raise ValueError(f"key '{key}' does not exist")
        item = loadable_item.load(**kwargs)
        if key in self.activated:
            return item
        load_times = {k: v.load_time for k, v in self.activated.items()}
        earliest_key = list(sort_dict_by_value(load_times).keys())[0]
        self.activated.pop(earliest_key).unload()
        self.activated[key] = loadable_item

        time_format = "-".join(TIME_FORMAT.split("-")[:-1])
        print(
            f"'{earliest_key}' is unloaded to make room for '{key}' "
            f"(last updated: {time.strftime(time_format, time.localtime(loadable_item.load_time))})"
        )
        return item
