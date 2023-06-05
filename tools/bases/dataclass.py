from abc import ABC
from dataclasses import fields, asdict, Field
from typing import Any, Dict, List, TypeVar

from tools.utils.icopy import shallow_copy_dict
from tools.utils.update import update_dict


TDataClass = TypeVar("TDataClass", bound="DataClassBase")


# 重建了DataClassBase抽象基类
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