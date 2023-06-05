import torch

from models.data.data_loader import IDataLoader
from tools.utils.to_type import np_batch_to_tensor
from tools.utils.type import tensor_dict_type
from tools.utils.device import to_device


class TensorBatcher:
    def __init__(self, loader: IDataLoader, device: torch.device) -> None:
        self.loader = loader
        self.device = device

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self) -> "TensorBatcher":
        self.loader.__iter__()
        return self

    def __next__(self) -> tensor_dict_type:
        npd = self.loader.__next__()
        batch = np_batch_to_tensor(npd)
        return to_device(batch, self.device)

    def to(self, device: torch.device) -> None:
        self.device = device

    def get_full_batch(self) -> tensor_dict_type:
        return np_batch_to_tensor(self.loader.get_full_batch())