import os
import shutil

from zipfile import ZipFile
from tempfile import mkdtemp
from typing import Any, Optional, ContextManager


def get_workspace(folder: str, *, force_new: bool = False) -> ContextManager:
    class _:
        tmp_folder: Optional[str]

        def __init__(self) -> None:
            self.tmp_folder = None

        def __enter__(self) -> str:
            if os.path.isdir(folder):
                if not force_new:
                    return folder
                self.tmp_folder = mkdtemp()
                shutil.copytree(folder, self.tmp_folder, dirs_exist_ok=True)
                return self.tmp_folder
            path = f"{folder}.zip"
            if not os.path.isfile(path):
                raise ValueError(f"neither '{folder}' nor '{path}' exists")
            self.tmp_folder = mkdtemp()
            with ZipFile(path, "r") as ref:
                ref.extractall(self.tmp_folder)
            return self.tmp_folder

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            if self.tmp_folder is not None:
                shutil.rmtree(self.tmp_folder)

    return _()