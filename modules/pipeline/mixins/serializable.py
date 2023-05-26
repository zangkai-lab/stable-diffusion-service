import json
import numpy as np

from typing import Generic, TypeVar, Dict, Union, Any, Type, List
from abc import ABCMeta, abstractmethod, ABC
from dataclasses import dataclass, fields, asdict, Field

from .withRegister import WithRegister
from ..utils import shallow_copy_dict, update_dict


np_dict_type = Dict[str, Union[np.ndarray, Any]]
TSArrays = TypeVar("TSArrays", bound="ISerializableArrays", covariant=True)
TSerializableArrays = TypeVar(
    "TSerializableArrays",
    bound="ISerializableArrays",
    covariant=True,
)
TSerializable = TypeVar("TSerializable", bound="ISerializable", covariant=True)
TDataClass = TypeVar("TDataClass", bound="DataClassBase")


class DataClassBase(ABC):
    @property
    def fields(self) -> List[Field]:
        return fields(self)

    @property
    def field_names(self) -> List[str]:
        return [f.name for f in self.fields]

    @property
    def attributes(self) -> List[Any]:
        return [getattr(self, name) for name in self.field_names]

    def asdict(self) -> Dict[str, Any]:
        return asdict(self)

    def copy(self: TDataClass) -> TDataClass:
        return type(self)(**shallow_copy_dict(asdict(self)))

    def update_with(self: TDataClass, other: TDataClass) -> TDataClass:
        d = update_dict(other.asdict(), self.asdict())
        return self.__class__(**d)


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

    def to_pack(self) -> JsonPack:
        return JsonPack(self.__identifier__, self.to_info())

    @classmethod
    def from_pack(cls: Type[TSerializable], pack: Dict[str, Any]) -> TSerializable:
        obj: ISerializable = cls.make(pack["type"], {})
        obj.from_info(pack["info"])
        obj.after_load()
        return obj

    def to_json(self) -> str:
        return json.dumps(self.to_pack().asdict())

    @classmethod
    def from_json(cls: Type[TSerializable], json_string: str) -> TSerializable:
        return cls.from_pack(json.loads(json_string))

    def copy(self: TSerializable) -> TSerializable:
        copied = self.__class__()
        copied.from_info(shallow_copy_dict(self.to_info()))
        return copied


class ISerializableArrays(ISerializable, Generic[TSArrays], metaclass=ABCMeta):
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