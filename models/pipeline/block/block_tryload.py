import os

from abc import ABCMeta, abstractmethod

from models.pipeline.pipeline_base import Block
from models.config.train_config import DLConfig


class TryLoadBlock(Block, metaclass=ABCMeta):
    # abstract

    @abstractmethod
    def try_load(self, folder: str) -> bool:
        pass

    @abstractmethod
    def from_scratch(self, config: DLConfig) -> None:
        pass

    @abstractmethod
    def dump_to(self, folder: str) -> None:
        pass

    # inheritance

    def build(self, config: DLConfig) -> None:
        if self.serialize_folder is not None:
            serialize_folder = os.path.join(self.serialize_folder, self.__identifier__)
            if self.try_load(serialize_folder):
                return
        self.from_scratch(config)

    def save_extra(self, folder: str) -> None:
        os.makedirs(folder, exist_ok=True)
        self.dump_to(folder)