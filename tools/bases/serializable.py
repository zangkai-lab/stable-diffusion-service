import json

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, Type, TypeVar, Callable

from tools.bases.register import WithRegister
from tools.bases.register import register_core
from tools.bases.register import TRegister
from tools.bases.dataclass import DataClassBase
from tools.utils.copy import shallow_copy_dict
from tools.utils.type import np_dict_type


TSerializable = TypeVar("TSerializable", bound="ISerializable", covariant=True)
TSerializableArrays = TypeVar(
    "TSerializableArrays",
    bound="ISerializableArrays",
    covariant=True,
)
TSDataClass = TypeVar("TSDataClass", bound="ISerializableDataClass", covariant=True)


@dataclass
class JsonPack(DataClassBase):
    type: str
    info: Dict[str, Any]


class ISerializable(WithRegister, Generic[TSerializable], metaclass=ABCMeta):
    # abstract

    @abstractmethod
    def to_info(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def from_info(self, info: Dict[str, Any]) -> None:
        pass

    # optional callbacks

    def after_load(self) -> None:
        pass

    # api
    # 序列化为JsonPack
    def to_pack(self) -> JsonPack:
        return JsonPack(self.__identifier__, self.to_info())

    # 工厂方法模式,反序列化pack创建对象
    @classmethod
    def from_pack(cls: Type[TSerializable], pack: Dict[str, Any]) -> TSerializable:
        obj: ISerializable = cls.make(pack["type"], {})
        obj.from_info(pack["info"])
        obj.after_load()
        return obj

    # 序列化为json字符串
    def to_json(self) -> str:
        return json.dumps(self.to_pack().asdict())

    # 工厂方法模式,反序列化json字符串创建对象
    @classmethod
    def from_json(cls: Type[TSerializable], json_string: str) -> TSerializable:
        return cls.from_pack(json.loads(json_string))

    def copy(self: TSerializable) -> TSerializable:
        copied = self.__class__()  # self.__class__()创建一个新的实例
        copied.from_info(shallow_copy_dict(self.to_info()))
        return copied


# 用于序列化的数据类, 依赖于注册类
@dataclass
class ISerializableDataClass(
    ISerializable,
    DataClassBase,
    Generic[TSDataClass],
    metaclass=ABCMeta,
):
    @classmethod
    @abstractmethod
    def d(cls) -> Dict[str, Type["ISerializableDataClass"]]:
        pass

    def to_info(self) -> Dict[str, Any]:
        return self.asdict()

    def from_info(self, info: Dict[str, Any]) -> None:
        for k, v in info.items():
            setattr(self, k, v)

    @classmethod
    def get(cls: Type[TRegister], name: str) -> Type[TRegister]:
        return cls.d()[name]

    @classmethod
    def has(cls, name: str) -> bool:
        return name in cls.d()

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
            cls.d(),
            allow_duplicate=allow_duplicate,
            before_register=before,
        )

    @classmethod
    def remove(cls, name: str) -> Callable:
        return cls.d().pop(name)

    @classmethod
    def check_subclass(cls, name: str) -> bool:
        return issubclass(cls.d()[name], cls)


class ISerializableArrays(ISerializable, Generic[TSerializableArrays], metaclass=ABCMeta):
    @abstractmethod
    def to_npd(self) -> np_dict_type:
        pass

    @abstractmethod
    def from_npd(self, npd: np_dict_type) -> None:
        pass

    def copy(self: TSerializableArrays) -> TSerializableArrays:
        copied: TSerializableArrays = super().copy()
        copied.from_npd(shallow_copy_dict(self.to_npd()))
        return copied