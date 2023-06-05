import numpy as np
import torch

from typing import Optional, Union, List, Any
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from torch.utils.data import SequentialSampler
from torch.utils.data.dataloader import _BaseDataLoaderIter
from torch.utils.data.distributed import DistributedSampler

from models.data.data_loader import IDataLoader
from models.data.dataset import IDataset
from models.data.data_processor import DataProcessor

from tools.utils.type import np_dict_type
from tools.utils.to_type import to_numpy
from tools.utils.ddp import get_world_size, get_ddp_info


class TorchDataset(IDataset):
    def __init__(self, dataset: IDataset, processor: DataProcessor) -> None:
        self.dataset = dataset
        self.processor = processor

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, item: Union[int, List[int], np.ndarray]) -> np_dict_type:
        batch = self.dataset[item]
        batch = self.processor.postprocess_item(batch)
        return batch


class _DataLoader(DataLoader):
    def __init__(
        self,
        dataset: TorchDataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        sampler: Optional[Sampler[int]] = None,
        *,
        use_distributed_sampler: Optional[bool] = None,
        **kwargs: Any,
    ):
        if use_distributed_sampler is None:
            use_distributed_sampler = get_ddp_info() is not None
        if use_distributed_sampler:
            if sampler is not None and not isinstance(sampler, DistributedSampler):
                raise ValueError(
                    "`sampler` should be `DistributedSampler` "
                    "when `use_distributed_sampler` is True"
                )
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            shuffle = False
        super().__init__(dataset, batch_size, shuffle, sampler, **kwargs)

    def __setattr__(self, attr: str, val: Any) -> None:
        if self.__initialized and attr in (
            "batch_size",
            "batch_sampler",
            "drop_last",
            "dataset",
            "persistent_workers",
        ):
            raise ValueError(
                f"{attr} attribute should not be set after "
                f"{self.__class__.__name__} is initialized"
            )

        super(DataLoader, self).__setattr__(attr, val)


class TorchDataLoader(IDataLoader):
    dataset: TorchDataset

    def __init__(
        self,
        loader: _DataLoader,
        *,
        sample_weights: Optional[np.ndarray] = None,
    ):
        if sample_weights is not None:
            raise ValueError(
                "in `DLLoader`, we should introduce `sample_weights` to the original "
                "Pytorch `DataLoader` (by specifying corresponding samplers)"
            )
        super().__init__(sample_weights=sample_weights)
        self.loader = loader
        self.use_numpy = False
        self.dataset = loader.dataset  # type: ignore
        self.sampler_backup = loader.sampler
        self._iterator: Optional[_BaseDataLoaderIter] = None

    def __iter__(self) -> "TorchDataLoader":
        self._iterator = self.loader.__iter__()
        return self

    def __next__(self) -> np_dict_type:
        if self._iterator is None:
            raise StopIteration
        batch = self._iterator.__next__()
        batch = {
            k: to_numpy(v) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        return batch

    def __len__(self) -> int:
        return len(self.loader)

    @property
    def batch_size(self) -> int:  # type: ignore
        return self.loader.batch_size * get_world_size()

    def copy(self) -> "TorchDataLoader":
        dataset = self.dataset
        self.__dict__.pop("dataset")
        copied = super().copy()
        assert isinstance(copied, TorchDataLoader)
        self.dataset = copied.dataset = dataset
        return copied

    def disable_shuffle(self) -> None:
        sampler = SequentialSampler(self.dataset)
        self.loader.sampler = sampler
        if hasattr(self.loader, "batch_sampler"):
            self.loader.batch_sampler.sampler = sampler

    def recover_shuffle(self) -> None:
        self.loader.sampler = self.sampler_backup
        if hasattr(self.loader, "batch_sampler"):
            self.loader.batch_sampler.sampler = self.sampler_backup