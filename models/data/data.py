
import numpy as np

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Generic, Optional, Type, TypeVar

from tools.bases.serializable import ISerializableArrays
from tools.utils.type import np_dict_type


data_dict: Dict[str, Type["IData"]] = {}

TData = TypeVar("TData", bound="IData", covariant=True)


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