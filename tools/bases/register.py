from typing import Any, Callable, Dict, List, Type, Union, TypeVar, Generic, Optional

from tools.utils.copy import shallow_copy_dict
from tools.utils.type import configs_type
from tools.utils.safe import safe_execute


TRegister = TypeVar("TRegister", bound="WithRegister", covariant=True)


# 注册的核心函数
def register_core(
    name: str,
    global_dict: Dict[str, type],
    *,
    allow_duplicate: bool = False,
    before_register: Optional[Callable] = None,
    after_register: Optional[Callable] = None,
):
    def _register(cls):
        if before_register is not None:
            before_register(cls)
        registered = global_dict.get(name)
        if registered is not None and not allow_duplicate:
            print(
                f"'{name}' has already registered "
                f"in the given global dict ({global_dict})"
            )
            return cls
        global_dict[name] = cls
        if after_register is not None:
            after_register(cls)
        return cls

    return _register


# 工厂类
class WithRegister(Generic[TRegister]):
    d: Dict[str, Type[TRegister]]
    __identifier__: str

    # 查看对应名字的类
    @classmethod
    def get(cls: Type[TRegister], name: str) -> Type[TRegister]:
        return cls.d[name]

    # 名字是否被注册
    @classmethod
    def has(cls, name: str) -> bool:
        return name in cls.d

    # 获取注册过的类，并创建实例
    @classmethod
    def make(
        cls: Type[TRegister],
        name: str,
        config: Dict[str, Any],
        *,
        ensure_safe: bool = False,
    ) -> TRegister:
        base = cls.get(name)
        if not ensure_safe:
            return base(**config)  # type: ignore
        return safe_execute(base, config)

    # 创建多个实例
    @classmethod
    def make_multiple(
        cls: Type[TRegister],
        names: Union[str, List[str]],
        configs: configs_type = None,
        *,
        ensure_safe: bool = False,
    ) -> List[TRegister]:
        if configs is None:
            configs = {}
        if isinstance(names, str):
            assert isinstance(configs, dict)
            return cls.make(names, configs, ensure_safe=ensure_safe)  # type: ignore
        if not isinstance(configs, list):
            configs = [configs.get(name, {}) for name in names]
        return [
            cls.make(name, shallow_copy_dict(config), ensure_safe=ensure_safe)
            for name, config in zip(names, configs)
        ]

    # 注册类
    @classmethod
    def register(
        cls,
        name: str,
        *,
        allow_duplicate: bool = False,
    ) -> Callable:
        def before(cls_: Type) -> None:
            cls_.__identifier__ = name

        return register_core(
            name,
            cls.d,
            allow_duplicate=allow_duplicate,
            before_register=before,
        )

    # 注册类
    @classmethod
    def remove(cls, name: str) -> Callable:
        return cls.d.pop(name)

    # 检查是否是WithRegister类本身的子类
    @classmethod
    def check_subclass(cls, name: str) -> bool:
        return issubclass(cls.d[name], cls)