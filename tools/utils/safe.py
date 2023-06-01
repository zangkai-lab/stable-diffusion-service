"""
inspect是Python的内建模块，它提供了很多有用的函数来帮助获取对象的信息，比如模块、类、函数、追踪记录、帧以及协程。
这些信息通常包括类的成员、文档字符串、源代码、规格以及函数的参数等
"""
import inspect

from typing import Any, Callable, Dict, Protocol, TypeVar


TFnResponse = TypeVar("TFnResponse")


# Desc: safe execute function
def check_requires(fn: Any, name: str, strict: bool = True) -> bool:
    if isinstance(fn, type):
        fn = fn.__init__  # type: ignore
    signature = inspect.signature(fn)
    for k, param in signature.parameters.items():
        if not strict and param.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if k == name:
            if param.kind is inspect.Parameter.VAR_POSITIONAL:
                return False
            return True
    return False


def filter_kw(
    fn: Callable,
    kwargs: Dict[str, Any],
    *,
    strict: bool = False,
) -> Dict[str, Any]:
    kw = {}
    for k, v in kwargs.items():
        if check_requires(fn, k, strict):
            kw[k] = v
    return kw


class Fn(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> TFnResponse:
        pass


def safe_execute(fn: Fn, kw: Dict[str, Any], *, strict: bool = False) -> TFnResponse:
    return fn(**filter_kw(fn, kw, strict=strict))