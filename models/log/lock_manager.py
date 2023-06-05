import inspect
import logging
import errno
import random
import shutil
import threading
import os
import time

from tools.handler.context import context_error_handler

from models.log.loggingmixin import LoggingMixin, timestamp


class _lock_file_refresher(threading.Thread):
    def __init__(self, lock_file, delay=1, refresh=0.01):
        super().__init__()
        self.__stop_event = threading.Event()
        self._lock_file, self._delay, self._refresh = lock_file, delay, refresh
        with open(lock_file, "r") as f:
            self._lock_file_contents = f.read()

    def run(self) -> None:
        counter = 0
        while True:
            counter += 1
            time.sleep(self._refresh)
            if counter * self._refresh >= self._delay:
                counter = 0
                with open(self._lock_file, "w") as f:
                    prefix = "\n\n"
                    add_line = f"{prefix}refreshed at {timestamp()}"
                    f.write(self._lock_file_contents + add_line)
            if self.__stop_event.is_set():
                break

    def stop(self):
        self.__stop_event.set()


class lock_manager(context_error_handler, LoggingMixin):
    """
    Util class to make simultaneously-write process safe with some
    hacked (ugly) tricks.
    """

    delay = 0.01
    __lock__ = "__lock__"

    def __init__(
            self,
            workspace,
            stuffs,
            verbose_level=None,
            set_lock=True,
            clear_stuffs_after_exc=True,
            name=None,
            wait=1000,
    ):
        self._workspace = workspace
        self._verbose_level = verbose_level
        self._name, self._wait = name, wait
        os.makedirs(workspace, exist_ok=True)
        self._stuffs, self._set_lock = stuffs, set_lock
        self._clear_stuffs = clear_stuffs_after_exc
        self._is_locked = False

    def __enter__(self):
        frame = inspect.currentframe().f_back
        self.log_msg(
            f"waiting for lock at {self.lock_file}",
            self.info_prefix,
            5,
            logging.DEBUG,
            frame,
        )
        enter_time = file_modify = None
        while True:
            try:
                fd = os.open(self.lock_file, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                self.log_msg(
                    "lock acquired",
                    self.info_prefix,
                    5,
                    logging.DEBUG,
                    frame,
                )
                if not self._set_lock:
                    self.log_msg(
                        "releasing lock since set_lock=False",
                        self.info_prefix,
                        5,
                        logging.DEBUG,
                        frame,
                    )
                    os.unlink(self.lock_file)
                    self.__refresher = None
                else:
                    self.log_msg(
                        "writing info to lock file",
                        self.info_prefix,
                        5,
                        logging.DEBUG,
                        frame,
                    )
                    with os.fdopen(fd, "a") as f:
                        f.write(
                            f"name      : {self._name}\n"
                            f"timestamp : {timestamp()}\n"
                            f"workspace : {self._workspace}\n"
                            f"stuffs    :\n{self.cache_stuffs_str}"
                        )
                    self.__refresher = _lock_file_refresher(self.lock_file)
                    self.__refresher.start()
                break
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
                try:
                    if file_modify is None:
                        enter_time = time.time()
                        file_modify = os.path.getmtime(self.lock_file)
                    else:
                        new_file_modify = os.path.getmtime(self.lock_file)
                        if new_file_modify != file_modify:
                            enter_time = time.time()
                            file_modify = new_file_modify
                        else:
                            wait_time = time.time() - enter_time
                            if wait_time >= self._wait:
                                raise ValueError(
                                    f"'{self.lock_file}' has been waited "
                                    f"for too long ({wait_time})"
                                )
                    time.sleep(random.random() * self.delay + self.delay)
                except ValueError:
                    msg = f"lock_manager was blocked by dead lock ({self.lock_file})"
                    self.exception(msg)
                    raise
                except FileNotFoundError:
                    pass
        self.log_block_msg(
            self.cache_stuffs_str,
            title="start processing following stuffs:",
            verbose_level=5,
            msg_level=logging.DEBUG,
            frame=frame,
        )
        self._is_locked = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.__refresher is not None:
            self.__refresher.stop()
            self.__refresher.join()
        if self._set_lock:
            super().__exit__(exc_type, exc_val, exc_tb)

    def _normal_exit(self, exc_type, exc_val, exc_tb, frame=None):
        if self._set_lock:
            os.unlink(self.lock_file)
        if frame is None:
            frame = inspect.currentframe().f_back.f_back.f_back
        self.log_msg("lock released", self.info_prefix, 5, logging.DEBUG, frame)

    def _exception_exit(self, exc_type, exc_val, exc_tb):
        frame = inspect.currentframe().f_back.f_back.f_back
        if self._clear_stuffs:
            for stuff in self._stuffs:
                if os.path.isfile(stuff):
                    self.log_msg(
                        f"clearing cached file: {stuff}",
                        ">> ",
                        5,
                        logging.ERROR,
                        frame,
                    )
                    os.remove(stuff)
                elif os.path.isdir(stuff):
                    self.log_msg(
                        f"clearing cached directory: {stuff}",
                        ">> ",
                        5,
                        logging.ERROR,
                        frame,
                    )
                    shutil.rmtree(stuff)
        self._normal_exit(exc_type, exc_val, exc_tb, frame)

    @property
    def locked(self):
        return self._is_locked

    @property
    def available(self):
        return not os.path.isfile(self.lock_file)

    @property
    def cache_stuffs_str(self):
        return "\n".join([f">> {stuff}" for stuff in self._stuffs])

    @property
    def exception_suffix(self):
        return f", clearing caches for safety{self.logging_suffix}"

    @property
    def lock_file(self):
        return os.path.join(self._workspace, self.__lock__)

    @property
    def logging_suffix(self):
        return "" if self._name is None else f" - {self._name}"