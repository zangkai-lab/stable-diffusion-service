import os
import json
import dill
import shutil
import zipfile
import numpy as np
import logging

from typing import Union, Optional
from functools import partial

from tools.utils.icopy import shallow_copy_dict
from tools.handler.context import context_error_handler

from models.log.loggingmixin import LoggingMixin
from models.log.lock_manager import lock_manager


class Saving(LoggingMixin):
    """
    Util class for saving instances.

    Methods
    ----------
    save_instance(instance, folder, log_method=None)
        Save instance to `folder`.
        * instance : object, instance to save.
        * folder : str, folder to save to.
        * log_method : {None, function}, used as `log_method` parameter in
        `log_with_external_method` method of `LoggingMixin`.

    load_instance(instance, folder, log_method=None)
        Load instance from `folder`.
        * instance : object, instance to load, need to be initialized.
        * folder : str, folder to load from.
        * log_method : {None, function}, used as `log_method` parameter in
        `log_with_external_method` method of `LoggingMixin`.

    """

    delim = "^_^"
    dill_suffix = ".pkl"
    array_sub_folder = "__arrays"

    @staticmethod
    def _check_core(elem):
        if isinstance(elem, dict):
            if not Saving._check_dict(elem):
                return False
        if isinstance(elem, (list, tuple)):
            if not Saving._check_list_and_tuple(elem):
                return False
        if not Saving._check_elem(elem):
            return False
        return True

    @staticmethod
    def _check_elem(elem):
        if isinstance(elem, (type, np.generic, np.ndarray)):
            return False
        if callable(elem):
            return False
        try:
            json.dumps({"": elem})
            return True
        except TypeError:
            return False

    @staticmethod
    def _check_list_and_tuple(arr: Union[list, tuple]):
        for elem in arr:
            if not Saving._check_core(elem):
                return False
        return True

    @staticmethod
    def _check_dict(d: dict):
        for v in d.values():
            if not Saving._check_core(v):
                return False
        return True

    @staticmethod
    def save_dict(d: dict, name: str, folder: str) -> str:
        if Saving._check_dict(d):
            kwargs = {}
            suffix, method, mode = ".json", json.dump, "w"
        else:
            kwargs = {"recurse": True}
            suffix, method, mode = Saving.dill_suffix, dill.dump, "wb"
        file = os.path.join(folder, f"{name}{suffix}")
        with open(file, mode) as f:
            method(d, f, **kwargs)
        return os.path.abspath(file)

    @staticmethod
    def load_dict(name: str, folder: str = None):
        if folder is None:
            folder, name = os.path.split(name)
        name, suffix = os.path.splitext(name)
        if not suffix:
            json_file = os.path.join(folder, f"{name}.json")
            if os.path.isfile(json_file):
                with open(json_file, "r") as f:
                    return json.load(f)
            dill_file = os.path.join(folder, f"{name}{Saving.dill_suffix}")
            if os.path.isfile(dill_file):
                with open(dill_file, "rb") as f:
                    return dill.load(f)
        else:
            assert_msg = f"suffix should be either 'json' or 'pkl', {suffix} found"
            assert suffix in {".json", ".pkl"}, assert_msg
            name = f"{name}{suffix}"
            file = os.path.join(folder, name)
            if os.path.isfile(file):
                if suffix == ".json":
                    mode, load_method = "r", json.load
                else:
                    mode, load_method = "rb", dill.load
                with open(file, mode) as f:
                    return load_method(f)
        raise ValueError(f"config '{name}' is not found under '{folder}' folder")

    @staticmethod
    def deep_copy_dict(d: dict):
        tmp_folder = os.path.join(os.getcwd(), "___tmp_dict_cache___")
        if os.path.isdir(tmp_folder):
            shutil.rmtree(tmp_folder)
        os.makedirs(tmp_folder)
        dict_name = "deep_copy"
        Saving.save_dict(d, dict_name, tmp_folder)
        loaded_dict = Saving.load_dict(dict_name, tmp_folder)
        shutil.rmtree(tmp_folder)
        return loaded_dict

    @staticmethod
    def get_cache_file(instance):
        return f"{type(instance).__name__}.pkl"

    @staticmethod
    def save_instance(instance, folder, log_method=None):
        instance_str = str(instance)
        Saving.log_with_external_method(
            f"saving '{instance_str}' to '{folder}'",
            Saving.info_prefix,
            log_method,
            5,
        )

        def _record_array(k, v):
            extension = ".npy" if isinstance(v, np.ndarray) else ".lst"
            array_attribute_dict[f"{k}{extension}"] = v

        def _check_array(attr_key_, attr_value_, depth=0):
            if isinstance(attr_value_, dict):
                for k in list(attr_value_.keys()):
                    v = attr_value_[k]
                    extended_k = f"{attr_key_}{delim}{k}"
                    if isinstance(v, dict):
                        _check_array(extended_k, v, depth + 1)
                    elif isinstance(v, array_types):
                        _record_array(extended_k, v)
                        attr_value_.pop(k)
            if isinstance(attr_value_, array_types):
                _record_array(attr_key_, attr_value_)
                if depth == 0:
                    cache_excludes.add(attr_key_)

        main_file = Saving.get_cache_file(instance)
        instance_dict = shallow_copy_dict(instance.__dict__)
        verbose, cache_excludes = map(
            getattr,
            [instance] * 2,
            ["lock_verbose", "cache_excludes"],
            [False, set()],
        )
        if os.path.isdir(folder):
            if verbose:
                prefix = Saving.warning_prefix
                msg = f"'{folder}' will be cleaned up when saving '{instance_str}'"
                Saving.log_with_external_method(
                    msg, prefix, log_method, msg_level=logging.WARNING
                )
            shutil.rmtree(folder)
        save_path = os.path.join(folder, main_file)
        array_folder = os.path.join(folder, Saving.array_sub_folder)
        tuple(
            map(
                lambda folder_: os.makedirs(folder_, exist_ok=True),
                [folder, array_folder],
            )
        )
        sorted_attributes, array_attribute_dict = sorted(instance_dict), {}
        delim, array_types = Saving.delim, (list, np.ndarray)
        for attr_key in sorted_attributes:
            if attr_key in cache_excludes:
                continue
            attr_value = instance_dict[attr_key]
            _check_array(attr_key, attr_value)
        cache_excludes.add("_verbose_level_")
        with lock_manager(
            folder,
            [os.path.join(folder, main_file)],
            name=instance_str,
        ):
            with open(save_path, "wb") as f:
                d = {k: v for k, v in instance_dict.items() if k not in cache_excludes}
                dill.dump(d, f, recurse=True)
        if array_attribute_dict:
            sorted_array_files = sorted(array_attribute_dict)
            sorted_array_files_full_path = list(
                map(lambda f_: os.path.join(array_folder, f_), sorted_array_files)
            )
            with lock_manager(
                array_folder,
                sorted_array_files_full_path,
                name=f"{instance_str} (arrays)",
            ):
                for array_file, array_file_full_path in zip(
                    sorted_array_files, sorted_array_files_full_path
                ):
                    array_value = array_attribute_dict[array_file]
                    if array_file.endswith(".npy"):
                        np.save(array_file_full_path, array_value)
                    elif array_file.endswith(".lst"):
                        with open(array_file_full_path, "wb") as f:
                            np.save(f, array_value)
                    else:
                        raise ValueError(
                            f"unrecognized file type '{array_file}' occurred"
                        )

    @staticmethod
    def load_instance(instance, folder, *, log_method=None, verbose=True):
        if verbose:
            Saving.log_with_external_method(
                f"loading '{instance}' from '{folder}'",
                Saving.info_prefix,
                log_method,
                5,
            )
        with open(os.path.join(folder, Saving.get_cache_file(instance)), "rb") as f:
            instance.__dict__.update(dill.load(f))
        delim = Saving.delim
        array_folder = os.path.join(folder, Saving.array_sub_folder)
        for array_file in os.listdir(array_folder):
            attr_name, attr_ext = os.path.splitext(array_file)
            if attr_ext == ".npy":
                load_method = partial(np.load, allow_pickle=True)
            elif attr_ext == ".lst":

                def load_method(path):
                    return np.load(path, allow_pickle=True).tolist()

            else:
                raise ValueError(f"unrecognized file type '{array_file}' occurred")
            array_value = load_method(os.path.join(array_folder, array_file))
            attr_hierarchy = attr_name.split(delim)
            if len(attr_hierarchy) == 1:
                instance.__dict__[attr_name] = array_value
            else:
                hierarchy_dict = instance.__dict__
                for attr in attr_hierarchy[:-1]:
                    hierarchy_dict = hierarchy_dict.setdefault(attr, {})
                hierarchy_dict[attr_hierarchy[-1]] = array_value

    @staticmethod
    def prepare_folder(instance, folder):
        if os.path.isdir(folder):
            instance.log_msg(
                f"'{folder}' already exists, it will be cleared up to save our model",
                instance.warning_prefix,
                msg_level=logging.WARNING,
            )
            shutil.rmtree(folder)
        os.makedirs(folder)

    @staticmethod
    def compress(abs_folder, remove_original=True):
        shutil.make_archive(abs_folder, "zip", abs_folder)
        if remove_original:
            shutil.rmtree(abs_folder)

    @staticmethod
    def compress_loader(
        folder: str,
        is_compress: bool,
        *,
        remove_extracted: bool = True,
        logging_mixin: Optional[LoggingMixin] = None,
    ):
        class _manager(context_error_handler):
            def __enter__(self):
                if is_compress:
                    if os.path.isdir(folder):
                        msg = (
                            f"'{folder}' already exists, "
                            "it will be cleared up to load our model"
                        )
                        if logging_mixin is None:
                            print(msg)
                        else:
                            logging_mixin.log_msg(
                                msg,
                                logging_mixin.warning_prefix,
                                msg_level=logging.WARNING,
                            )
                        shutil.rmtree(folder)
                    with zipfile.ZipFile(f"{folder}.zip", "r") as zip_ref:
                        zip_ref.extractall(folder)

            def _normal_exit(self, exc_type, exc_val, exc_tb):
                if is_compress and remove_extracted:
                    shutil.rmtree(folder)

        return _manager()