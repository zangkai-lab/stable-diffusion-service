import json

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, Type, TypeVar

from tools.bases.register import WithRegister
from tools.bases.dataclass import DataClassBase
from tools.utils.copy import shallow_copy_dict


TSerializable = TypeVar("TSerializable", bound="ISerializable", covariant=True)


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