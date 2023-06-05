import os
import json
import numpy as np

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, Type, TypeVar, Callable, Optional

from tools.bases.register import WithRegister
from tools.bases.register import register_core
from tools.bases.register import TRegister
from tools.bases.dataclass import DataClassBase
from tools.utils.icopy import shallow_copy_dict
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


class Serializer:
    id_file: str = "id.txt"
    info_file: str = "info.json"
    npd_folder: str = "npd"

    @classmethod
    def save_info(
        cls,
        folder: str,
        *,
        info: Optional[Dict[str, Any]] = None,
        serializable: Optional[ISerializable] = None,
    ) -> None:
        os.makedirs(folder, exist_ok=True)
        if info is None and serializable is None:
            raise ValueError("either `info` or `serializable` should be provided")
        if info is None:
            info = serializable.to_info()
        with open(os.path.join(folder, cls.info_file), "w") as f:
            json.dump(info, f)

    @classmethod
    def load_info(cls, folder: str) -> Dict[str, Any]:
        return cls.try_load_info(folder, strict=True)

    @classmethod
    def try_load_info(
        cls,
        folder: str,
        *,
        strict: bool = False,
    ) -> Optional[Dict[str, Any]]:
        info_path = os.path.join(folder, cls.info_file)
        if not os.path.isfile(info_path):
            if not strict:
                return
            raise ValueError(f"'{info_path}' does not exist")
        with open(info_path, "r") as f:
            info = json.load(f)
        return info

    @classmethod
    def save_npd(
        cls,
        folder: str,
        *,
        npd: Optional[np_dict_type] = None,
        serializable: Optional[ISerializableArrays] = None,
    ) -> None:
        os.makedirs(folder, exist_ok=True)
        if npd is None and serializable is None:
            raise ValueError("either `npd` or `serializable` should be provided")
        if npd is None:
            npd = serializable.to_npd()
        npd_folder = os.path.join(folder, cls.npd_folder)
        os.makedirs(npd_folder, exist_ok=True)
        for k, v in npd.items():
            np.save(os.path.join(npd_folder, f"{k}.npy"), v)

    @classmethod
    def load_npd(cls, folder: str) -> np_dict_type:
        os.makedirs(folder, exist_ok=True)
        npd_folder = os.path.join(folder, cls.npd_folder)
        if not os.path.isdir(npd_folder):
            raise ValueError(f"'{npd_folder}' does not exist")
        npd = {}
        for file in os.listdir(npd_folder):
            key = os.path.splitext(file)[0]
            npd[key] = np.load(os.path.join(npd_folder, file))
        return npd

    @classmethod
    def save(
        cls,
        folder: str,
        serializable: ISerializable,
        *,
        save_npd: bool = True,
    ) -> None:
        cls.save_info(folder, serializable=serializable)
        if save_npd and isinstance(serializable, ISerializableArrays):
            cls.save_npd(folder, serializable=serializable)
        with open(os.path.join(folder, cls.id_file), "w") as f:
            f.write(serializable.__identifier__)

    @classmethod
    def load(
        cls,
        folder: str,
        base: Type[TSerializable],
        *,
        swap_id: Optional[str] = None,
        swap_info: Optional[Dict[str, Any]] = None,
        load_npd: bool = True,
    ) -> TSerializable:
        serializable = cls.load_empty(folder, base, swap_id=swap_id)
        serializable.from_info(swap_info or cls.load_info(folder))
        if load_npd and isinstance(serializable, ISerializableArrays):
            serializable.from_npd(cls.load_npd(folder))
        return serializable

    @classmethod
    def load_empty(
        cls,
        folder: str,
        base: Type[TSerializable],
        *,
        swap_id: Optional[str] = None,
    ) -> TSerializable:
        if swap_id is not None:
            s_type = swap_id
        else:
            id_path = os.path.join(folder, cls.id_file)
            if not os.path.isfile(id_path):
                raise ValueError(f"cannot find '{id_path}'")
            with open(id_path, "r") as f:
                s_type = f.read().strip()
        return base.make(s_type, {})