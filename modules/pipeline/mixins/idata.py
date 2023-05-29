import torch
import json
import numpy as np

from torch import Tensor
from abc import ABCMeta, abstractmethod, ABC
from typing import Dict, Any, Optional, Type, Generic, TypeVar, Union, List, NamedTuple, Tuple
from dataclasses import dataclass

from .serializable import ISerializableArrays
from .serializable import ISerializableDataClass
from .serializable import ISerializable
from .serializable import DataClassBase
from .iblock import IBlock

from ..core import IPipeline
from ..utils import to_numpy
from ..utils import to_torch
from ..utils import is_local_rank_0
from ..utils import safe_execute

data_dict: Dict[str, Type["IData"]] = {}
data_configs: Dict[str, Type["DataConfig"]] = {}
data_type = Optional[Union[np.ndarray, str]]
np_dict_type = Dict[str, Union[np.ndarray, Any]]
tensor_dict_type = Dict[str, Union[torch.Tensor, Any]]
TDataBundleItem = Optional[Union[data_type, np_dict_type, tensor_dict_type, Any]]
data_processor_configs: Dict[str, Type["DataProcessorConfig"]] = {}
sample_weights_type = Optional[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]
TSplitSW = Tuple[Optional[np.ndarray], Optional[np.ndarray]]

TData = TypeVar("TData", bound="IData", covariant=True)
TDataProcessor = TypeVar("TDataProcessor", bound="DataProcessor", covariant=True)
TDataLoaders = Tuple["IDataLoader", Optional["IDataLoader"]]


class IDataset(ABC):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, item: Union[int, List[int], np.ndarray]) -> Dict[str, Any]:
        pass


@dataclass
class DataConfig(ISerializableDataClass):
    for_inference: bool = False
    batch_size: int = 1
    valid_batch_size: Optional[int] = None
    shuffle_train: bool = True
    shuffle_valid: bool = False

    @classmethod
    def d(cls) -> Dict[str, Type["DataConfig"]]:
        return data_configs


class PureFromInfoMixin:
    def from_info(self, info: Dict[str, Any]) -> None:
        for k, v in info.items():
            setattr(self, k, v)


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


class IDataBlock(PureFromInfoMixin, IBlock, ISerializable, metaclass=ABCMeta):
    config: "DataProcessorConfig"
    previous: Dict[str, "IDataBlock"]

    def __init__(self, **kwargs: Any) -> None:
        not_exists_tag = "$$NOT_EXISTS$$"
        for field in self.fields:
            value = kwargs.get(field, not_exists_tag)
            if value == not_exists_tag:
                raise ValueError(
                    f"Argument '{field}' needs to be provided "
                    f"for `{self.__class__.__name__}`."
                )
            setattr(self, field, value)

    # inherit

    def build(self, config: "DataProcessorConfig") -> None:
        self.config = config
        configs = (config.block_configs or {}).setdefault(self.__identifier__, {})
        for field in self.fields:
            setattr(self, field, configs.setdefault(field, getattr(self, field)))

    def to_info(self) -> Dict[str, Any]:
        return {field: getattr(self, field) for field in self.fields}

    # abstract

    @property
    @abstractmethod
    def fields(self) -> List[str]:
        pass

    @abstractmethod
    def transform(self, bundle: DataBundle, for_inference: bool) -> DataBundle:
        """
        This method should not utilize `config`!

        Changes can happen inplace.
        """

    @abstractmethod
    def fit_transform(self, bundle: DataBundle) -> DataBundle:
        """
        This method should prepare necessary info, which might be used
        in the `to_info` method.

        If any necessary info comes from `config`, this method should extract
        them and assign them to the corresponding properties.

        This method will NOT be called in a loading procedure, and the
        necessary info should be loaded in the `from_info` method.

        This method will always assume `for_inference=False`.

        Changes can happen inplace.
        """
    # optional callbacks

    # changes can happen inplace
    def postprocess_item(self, item: Any) -> Any:
        return item

    # changes can happen inplace
    def recover_labels(self, y: np.ndarray) -> np.ndarray:
        return y

    # api

    @property
    def is_local_rank_0(self) -> bool:
        return is_local_rank_0()


class INoInitDataBlock(IDataBlock):
    @property
    def fields(self) -> List[str]:
        return []


@dataclass
class DataProcessorConfig(ISerializableDataClass):
    block_names: Optional[List[str]] = None
    block_configs: Optional[Dict[str, Dict[str, Any]]] = None

    @classmethod
    def d(cls) -> Dict[str, Type["DataProcessorConfig"]]:
        return data_processor_configs

    @property
    def default_blocks(self) -> List[IDataBlock]:
        return []

    def add_blocks(self, *blocks: IDataBlock) -> None:
        if self.block_names is None:
            self.block_names = []
        for b in blocks:
            b_id = b.__identifier__
            if b_id in self.block_names:
                print(f"block `{b_id}` already exists, it will be skipped")
            self.block_names.append(b_id)
            if isinstance(b, INoInitDataBlock):
                continue
            if self.block_configs is None:
                self.block_configs = {}
            self.block_configs[b_id] = b.to_info()

    def set_blocks(self, *blocks: IDataBlock) -> None:
        self.block_names = []
        self.add_blocks(*blocks)


@IPipeline.register("base.data_processor")
class DataProcessor(IPipeline):
    config: DataProcessorConfig
    blocks: List[IDataBlock]
    is_ready: bool = False

    # inheritance

    @classmethod
    def init(
        cls: Type[TDataProcessor],
        config: Optional[DataProcessorConfig],
    ) -> TDataProcessor:
        self: DataProcessor = cls()
        self.config = (config or self.config_base()).copy()
        if self.config.block_names is None:
            self.config.set_blocks(*self.config.default_blocks)
        self.before_build_in_init()
        self.build(*(IDataBlock.get(name)() for name in self.config.block_names))  # type: ignore
        return self

    # optional callbacks

    @property
    def config_base(self) -> Type[DataProcessorConfig]:
        return DataProcessorConfig

    @property
    def block_base(self) -> Type[IDataBlock]:
        return IDataBlock

    def before_build_in_init(self) -> None:
        pass

    def after_load(self) -> None:
        self.is_ready = True

    # api

    def _run(self, fn: str, bundle: DataBundle, for_inference: bool) -> DataBundle:
        kw = dict(bundle=bundle.copy(), for_inference=for_inference)
        previous: Dict[str, IDataBlock] = {}
        for block in self.blocks:
            block.previous = previous
            kw["bundle"] = safe_execute(getattr(block, fn), kw)
            previous[block.__identifier__] = block
        return kw["bundle"]  # type: ignore

    def transform(self, bundle: DataBundle, *, for_inference: bool) -> DataBundle:
        return self._run("transform", bundle, for_inference)

    def fit_transform(self, bundle: DataBundle) -> DataBundle:
        bundle = self._run("fit_transform", bundle, False)
        self.is_ready = True
        return bundle

    # changes can happen inplace
    def postprocess_item(self, item: Any) -> np_dict_type:
        for block in self.blocks:
            item = block.postprocess_item(item)
        return item

    def recover_labels(self, y: np.ndarray) -> np.ndarray:
        for block in self.blocks[::-1]:
            y = block.recover_labels(y)
        return y


def norm_sw(sample_weights: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if sample_weights is None:
        return None
    return sample_weights / sample_weights.sum()


def split_sw(sample_weights: sample_weights_type) -> TSplitSW:
    if sample_weights is None:
        train_weights = valid_weights = None
    else:
        if not isinstance(sample_weights, np.ndarray):
            train_weights, valid_weights = sample_weights
        else:
            train_weights, valid_weights = sample_weights, None
    train_weights, valid_weights = map(norm_sw, [train_weights, valid_weights])
    return train_weights, valid_weights


class IData(ISerializableArrays, Generic[TData], metaclass=ABCMeta):
    d = data_dict

    train_dataset: IDataset
    valid_dataset: Optional[IDataset]

    train_weights: Optional[np.ndarray]
    valid_weights: Optional[np.ndarray]

    config: DataConfig
    processor: DataProcessor
    bundle: Optional[DataBundle]

    for_inference: bool

    def __init__(self) -> None:
        self.train_weights = None
        self.valid_weights = None

    # abstract

    @abstractmethod
    def get_loaders(self) -> TDataLoaders:
        pass

    # inheritance

    def to_info(self) -> Dict[str, Any]:
        if not self.processor.is_ready:
            raise ValueError(
                "`processor` should be ready before calling `to_info`, "
                "did you forget to call the `fit` method first?"
            )
        return {
            "type": self.__identifier__,
            "processor": self.processor.to_pack().asdict(),
            "config": self.config.to_pack().asdict(),
            "bundle": None if self.bundle is None else self.bundle.to_info(),
        }

    def from_info(self, info: Dict[str, Any]) -> None:
        if self.__identifier__ != info["type"]:
            msg = f"type does not match: {self.__identifier__} != {info['type']}"
            raise ValueError(msg)
        self.processor = self.processor_base.from_pack(info["processor"])
        self.config = self.config_base.from_pack(info["config"])
        bundle_info = info["bundle"]
        if not bundle_info:
            self.bundle = None
        else:
            self.bundle = DataBundle.empty()
            self.bundle.from_info(bundle_info)

    def to_npd(self) -> np_dict_type:
        return {} if self.bundle is None else self.bundle.to_npd()

    def from_npd(self, npd: np_dict_type) -> None:
        if npd:
            if self.bundle is None:
                self.bundle = DataBundle.empty()
            self.bundle.from_npd(npd)

    # optional callback

    @property
    def config_base(self) -> Type[DataConfig]:
        return DataConfig

    @property
    def processor_base(self) -> Type[DataProcessor]:
        return DataProcessor

    def get_bundle(
        self,
        x_train: data_type,
        y_train: Optional[data_type] = None,
        x_valid: Optional[data_type] = None,
        y_valid: Optional[data_type] = None,
        train_others: Optional[np_dict_type] = None,
        valid_others: Optional[np_dict_type] = None,
        *args: Any,
        **kwargs: Any,
    ) -> DataBundle:
        args = x_train, y_train, x_valid, y_valid, train_others, valid_others
        return DataBundle(*args)

    def set_sample_weights(self: TData, sample_weights: sample_weights_type) -> TData:
        self.train_weights, self.valid_weights = split_sw(sample_weights)
        return self

    # api

    @classmethod
    def init(
        cls: Type[TData],
        config: Optional[DataConfig] = None,
        processor_config: Optional[DataProcessorConfig] = None,
    ) -> TData:
        self: TData = cls()
        self.bundle = None
        self.config = config or self.config_base()
        self.processor = self.processor_base.init(processor_config)
        return self

    def fit(
        self: TData,
        x_train: data_type,
        y_train: Optional[data_type] = None,
        x_valid: Optional[data_type] = None,
        y_valid: Optional[data_type] = None,
        train_others: Optional[np_dict_type] = None,
        valid_others: Optional[np_dict_type] = None,
        *args: Any,
        **kwargs: Any,
    ) -> TData:
        args = x_train, y_train, x_valid, y_valid, train_others, valid_others, *args
        bundle = self.get_bundle(*args, **kwargs)
        bundle = self.processor.fit_transform(bundle)
        self.bundle = bundle
        return self

    def transform(self, *args: Any, **kwargs: Any) -> DataBundle:
        if not self.processor.is_ready:
            raise ValueError("`processor` should be ready before calling `transform`")
        bundle = self.get_bundle(*args, **kwargs)
        bundle = self.processor.transform(bundle, for_inference=True)
        return bundle

    def recover_labels(self, y: np.ndarray) -> np.ndarray:
        return self.processor.recover_labels(y)
