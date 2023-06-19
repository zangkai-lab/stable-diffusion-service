import os
import json
import urllib.request
import hashlib
import tqdm

from typing import Optional, NamedTuple
from zipfile import ZipFile

from models.zoo.parameters import OPT


class DownloadProgressBar(tqdm):
    def update_to(
        self,
        b: int = 1,
        bsize: int = 1,
        tsize: Optional[int] = None,
    ) -> None:
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


class FileInfo(NamedTuple):
    sha: str
    st_size: int
    download_url: Optional[str] = None


def _get_file_size(path: str) -> int:
    return os.stat(path).st_size


def check_available(tag: str, repo: str, name: str) -> Optional[FileInfo]:
    with open(os.path.join(os.path.dirname(__file__), "available.json"), "r") as f:
        available = json.load(f)
    if repo != "pretrained-models":
        return FileInfo("", 0)
    info = available[tag].get(name)
    return None if info is None else FileInfo(**info)


def _check_sha(path: str, tgt_sha: str) -> bool:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest() == tgt_sha


def download(
    tag: str,
    repo: str,
    name: str,
    root: str,
    extension: str,
    *,
    check_sha: bool = False,
    remove_zip: bool = True,
) -> str:
    info = check_available(tag, repo, name)
    if info is None:
        raise ValueError(f"'{name}' is currently not available at '{tag}'")
    os.makedirs(root, exist_ok=True)
    file = f"{name}.{extension}"
    path = os.path.join(root, file)
    is_zip = extension == "zip"
    zip_folder_path = os.path.join(root, name)
    if is_zip and os.path.isdir(zip_folder_path):
        return zip_folder_path
    fmt = "cache file is detected but {}, it will be re-downloaded"
    if not is_zip and os.path.isfile(path):
        if os.stat(path).st_size != info.st_size:
            print(fmt.format("st_size is not correct"))
        else:
            if not check_sha or _check_sha(path, info.sha):
                return path
            print(fmt.format("sha is not correct"))
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=name) as t:
        if info.download_url is not None:
            url = info.download_url
        else:
            prefix = f"https://github.com/carefree0910/{repo}/releases/download/{tag}/"
            url = f"{prefix}{file}"
        urllib.request.urlretrieve(
            url,
            filename=path,
            reporthook=t.update_to,
        )
    if not is_zip:
        return path
    with ZipFile(path, "r") as zip_ref:
        zip_ref.extractall(zip_folder_path)
    if remove_zip:
        os.remove(path)
    return zip_folder_path


def download_model(
    name: str,
    *,
    root: Optional[str] = None,
) -> str:
    if root is None:
        root = os.path.join(OPT.cache_dir, "models")
    return download("checkpoints", "pretrained-models", name, root, "pt")


def download_static(
    name: str,
    *,
    extension: str,
    root: Optional[str] = None,
) -> str:
    if root is None:
        root = os.path.join(OPT.cache_dir, "static")
    return download("static", "pretrained-models", name, root, extension)