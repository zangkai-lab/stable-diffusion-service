import json
import numpy as np

from torch import Tensor
from typing import Any, Dict, List, Optional, Union, Tuple, NamedTuple
from dataclasses import dataclass

from tools.bases.dataclass import DataClassBase
from tools.utils.type import np_dict_type, tensor_dict_type, data_type
from tools.utils.to_type import to_torch, to_numpy


TDataBundleItem = Optional[Union[data_type, np_dict_type, tensor_dict_type, Any]]


class DataArgs(NamedTuple):
    x: TDataBundleItem
    y: TDataBundleItem
    others: Optional[np_dict_type]

    @property
    def xy(self) -> Tuple[TDataBundleItem, TDataBundleItem]:
        return self.x, self.y


def copy_data(data: TDataBundleItem) -> data_type:
    if data is None:
        return None
    if isinstance(data, dict):
        return {k: copy_data(v) for k, v in data.items()}
    if isinstance(data, np.ndarray):
        return data.copy()
    if isinstance(data, Tensor):
        return data.clone()
    return data


def check_data_is_info(data: TDataBundleItem) -> bool:
    if (
        data is None
        or isinstance(data, dict)
        or isinstance(data, np.ndarray)
        or isinstance(data, Tensor)
    ):
        return False
    try:
        json.dumps([data])
        return True
    except:
        return False


@dataclass
class DataBundle(DataClassBase):
    x_train: TDataBundleItem
    y_train: TDataBundleItem = None
    x_valid: TDataBundleItem = None
    y_valid: TDataBundleItem = None
    train_others: Optional[np_dict_type] = None
    valid_others: Optional[np_dict_type] = None

    @property
    def train_args(self) -> DataArgs:
        return DataArgs(self.x_train, self.y_train, self.train_others)

    @property
    def valid_args(self) -> DataArgs:
        return DataArgs(self.x_valid, self.y_valid, self.valid_others)

    def copy(self) -> "DataBundle":
        return DataBundle(*map(copy_data, self.attributes))

    def to_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        for k, v in self.asdict().items():
            if check_data_is_info(v):
                info[k] = v
        return info

    def from_info(self, info: Dict[str, Any]) -> None:
        for k, v in info.items():
            setattr(self, k, v)

    def to_npd(self) -> np_dict_type:
        def _to_np(key: str, data: Union[np.ndarray, Tensor]) -> np.ndarray:
            if isinstance(data, np.ndarray):
                return data
            tensor_keys.append(key)
            return to_numpy(data)

        npd: np_dict_type = {}
        tensor_keys: List[str] = []
        for k, v in self.asdict().items():
            if isinstance(v, dict):
                v = {f"{k}.{vk}": vv for vk, vv in v.items()}
                npd.update({vk: _to_np(vk, vv) for vk, vv in v.items()})
            elif isinstance(v, (np.ndarray, Tensor)):
                npd[k] = _to_np(k, v)
        if tensor_keys:
            npd["__tensor_keys__"] = np.array(tensor_keys)
        return npd

    def from_npd(self, npd: np_dict_type) -> None:
        attr_collections: Dict[str, Union[np_dict_type, tensor_dict_type]] = {}
        tensor_keys = set(npd.pop("__tensor_keys__", np.array([])).tolist())
        for k, v in npd.items():
            attr = None
            if "." in k:
                attr, k = k.split(".", 1)
            if k in tensor_keys:
                v = to_torch(v)
            if attr is None:
                setattr(self, k, v)
            else:
                attr_collections.setdefault(attr, {})[k] = v
        for attr, collection in attr_collections.items():
            setattr(self, attr, collection)

    @classmethod
    def empty(cls) -> "DataBundle":
        return cls(None)