import logging
import math
import sys
import inspect
import datetime
import os
import time

from typing import Optional

from tools.utils.fix import fix_float_to_length


class Incrementer:
    """
    Util class which can calculate running mean & running std efficiently.

    Parameters
    ----------
    window_size : {int, None}, window size of running statistics.
    * If None, then all history records will be used for calculation.
    """

    def __init__(self, window_size: int = None):
        if window_size is not None:
            if not isinstance(window_size, int):
                msg = f"window size should be integer, {type(window_size)} found"
                raise ValueError(msg)
            if window_size < 2:
                msg = f"window size should be greater than 2, {window_size} found"
                raise ValueError(msg)
        self._window_size = window_size
        self._n_record = self._previous = None
        self._running_sum = self._running_square_sum = None

    @property
    def mean(self):
        return self._running_sum / self._n_record

    @property
    def std(self):
        return math.sqrt(
            max(
                0.0,
                self._running_square_sum / self._n_record - self.mean ** 2,
            )
        )

    @property
    def n_record(self):
        return self._n_record

    def update(self, new_value):
        if self._n_record is None:
            self._n_record = 1
            self._running_sum = new_value
            self._running_square_sum = new_value ** 2
        else:
            self._n_record += 1
            self._running_sum += new_value
            self._running_square_sum += new_value ** 2
        if self._window_size is not None:
            if self._previous is None:
                self._previous = [new_value]
            else:
                self._previous.append(new_value)
            if self._n_record == self._window_size + 1:
                self._n_record -= 1
                previous = self._previous.pop(0)
                self._running_sum -= previous
                self._running_square_sum -= previous ** 2


class _Formatter(logging.Formatter):
    """Formatter for logging, which supports millisecond."""

    converter = datetime.fromtimestamp

    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            s = ct.strftime(datefmt)
        else:
            t = ct.strftime("%Y-%m-%d %H:%M:%S")
            s = "%s.%03d" % (t, record.msecs)
        return s

    def formatMessage(self, record: logging.LogRecord) -> str:
        record.__dict__.setdefault("func_prefix", "Unknown")
        return super().formatMessage(record)


def truncate_string_to_length(string: str, length: int) -> str:
    """Truncate a string to make sure its length not exceeding a given length."""

    if len(string) <= length:
        return string
    half_length = int(0.5 * length) - 1
    head = string[:half_length]
    tail = string[-half_length:]
    return f"{head}{'.' * (length - 2 * half_length)}{tail}"


def timestamp(simplify: bool = False, ensure_different: bool = False) -> str:
    """
    Return current timestamp.

    Parameters
    ----------
    simplify : bool. If True, format will be simplified to 'year-month-day'.
    ensure_different : bool. If True, format will include millisecond.

    Returns
    -------
    timestamp : str

    """

    now = datetime.now()
    if simplify:
        return now.strftime("%Y-%m-%d")
    if ensure_different:
        return now.strftime("%Y-%m-%d_%H-%M-%S-%f")
    return now.strftime("%Y-%m-%d_%H-%M-%S")


class LoggingMixin:
    """
    Mixin class to provide logging methods for base class.

    Attributes
    ----------
    _triggered_ : bool
    * If not `_triggered_`, log file will not be created.

    _verbose_level_ : int
    * Preset verbose level of the whole logging process.

    Methods
    ----------
    log_msg(self, body, prefix="", verbose_level=1)
        Log something either through console or to a file.
        * body : str
            Main logging message.
        * prefix : str
            Prefix added to `body` when logging message goes through console.
        * verbose_level : int
            If `self._verbose_level_` >= verbose_level, then logging message
            will go through console.

    log_block_msg(self, body, prefix="", title="", verbose_level=1)
        Almost the same as `log_msg`, except adding `title` on top of `body`.

    """

    _triggered_ = False
    _initialized_ = False
    _logging_path_ = None
    _logger_ = _verbose_level_ = None
    _date_format_string_ = "%Y-%m-%d %H:%M:%S.%f"
    _formatter_ = _Formatter(
        "[ {asctime:s} ] [ {levelname:^8s} ] {func_prefix:s} {message:s}",
        _date_format_string_,
        style="{",
    )
    _timing_dict_, _time_cache_dict_ = {}, {}

    info_prefix = ">  [ info ] "
    warning_prefix = "> [warning] "
    error_prefix = "> [ error ] "

    @property
    def logging_path(self):
        if self._logging_path_ is None:
            folder = os.path.join(os.getcwd(), "_logging", type(self).__name__)
            os.makedirs(folder, exist_ok=True)
            self._logging_path_ = self.generate_logging_path(folder)
        return self._logging_path_

    @property
    def console_handler(self):
        if self._logger_ is None:
            return
        for handler in self._logger_.handlers:
            if isinstance(handler, logging.StreamHandler):
                return handler

    @staticmethod
    def _get_func_prefix(frame=None, return_prefix=True):
        if frame is None:
            frame = inspect.currentframe().f_back.f_back
        if not return_prefix:
            return frame
        frame_info = inspect.getframeinfo(frame)
        file_name = truncate_string_to_length(os.path.basename(frame_info.filename), 16)
        func_name = truncate_string_to_length(frame_info.function, 24)
        func_prefix = (
            f"[ {func_name:^24s} ] [ {file_name:>16s}:{frame_info.lineno:<4d} ]"
        )
        return func_prefix

    @staticmethod
    def _release_handlers(logger):
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    @staticmethod
    def generate_logging_path(folder: str) -> str:
        return os.path.join(folder, f"{timestamp()}.log")

    def _init_logging(self, verbose_level: Optional[int] = 2, trigger: bool = True):
        wants_trigger = trigger and not LoggingMixin._triggered_
        if LoggingMixin._initialized_ and not wants_trigger:
            return self
        LoggingMixin._initialized_ = True
        logger_name = getattr(self, "_logger_name_", "root")
        logger = LoggingMixin._logger_ = logging.getLogger(logger_name)
        LoggingMixin._verbose_level_ = verbose_level
        if not trigger:
            return self
        LoggingMixin._triggered_ = True
        config = getattr(self, "config", {})
        self._logging_path_ = config.get("_logging_path_")
        if self._logging_path_ is None:
            self._logging_path_ = config["_logging_path_"] = self.logging_path
        os.makedirs(os.path.dirname(self.logging_path), exist_ok=True)
        file_handler = logging.FileHandler(self.logging_path, encoding="utf-8")
        file_handler.setFormatter(self._formatter_)
        file_handler.setLevel(logging.DEBUG)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(_Formatter("{custom_prefix:s}{message:s}", style="{"))
        logger.setLevel(logging.DEBUG)
        self._release_handlers(logger)
        logger.addHandler(console)
        logger.addHandler(file_handler)
        self.log_block_msg(sys.version, title="system version", verbose_level=None)
        return self

    def log_msg(
            self,
            body: str,
            prefix: str = "",
            verbose_level: Optional[int] = 1,
            msg_level: int = logging.INFO,
            frame=None,
    ):
        preset_verbose_level = getattr(self, "_verbose_level", None)
        if preset_verbose_level is not None:
            self._verbose_level_ = preset_verbose_level
        elif self._verbose_level_ is None:
            self._verbose_level_ = 0
        console_handler = self.console_handler
        if verbose_level is None or self._verbose_level_ < verbose_level:
            do_print, console_level = False, msg_level + 10
        else:
            do_print, console_level = not LoggingMixin._triggered_, msg_level
        if console_handler is not None:
            console_handler.setLevel(console_level)
        if do_print:
            print(prefix + body)
        elif LoggingMixin._triggered_:
            func_prefix = self._get_func_prefix(frame)
            self._logger_.log(
                msg_level,
                body,
                extra={"func_prefix": func_prefix, "custom_prefix": prefix},
            )
        if console_handler is not None:
            console_handler.setLevel(logging.INFO)

    def log_block_msg(
            self,
            body: str,
            prefix: str = "",
            title: str = "",
            verbose_level: Optional[int] = 1,
            msg_level: int = logging.INFO,
            frame=None,
    ):
        frame = self._get_func_prefix(frame, False)
        self.log_msg(f"{title}\n{body}\n", prefix, verbose_level, msg_level, frame)

    def exception(self, body, frame=None):
        self._logger_.exception(
            body,
            extra={
                "custom_prefix": self.error_prefix,
                "func_prefix": LoggingMixin._get_func_prefix(frame),
            },
        )

    @staticmethod
    def log_with_external_method(body, prefix, log_method, *args, **kwargs):
        if log_method is None:
            print(prefix + body)
        else:
            kwargs["frame"] = LoggingMixin._get_func_prefix(
                kwargs.pop("frame", None),
                False,
            )
            log_method(body, prefix, *args, **kwargs)

    @staticmethod
    def merge_logs_by_time(*log_files, tgt_file):
        tgt_folder = os.path.dirname(tgt_file)
        date_str_len = (
                len(datetime.today().strftime(LoggingMixin._date_format_string_)) + 4
        )
        with lock_manager(tgt_folder, [tgt_file], clear_stuffs_after_exc=False):
            msg_dict, msg_block, last_searched = {}, [], None
            for log_file in log_files:
                with open(log_file, "r") as f:
                    for line in f:
                        date_str = line[:date_str_len]
                        if date_str[:2] == "[ " and date_str[-2:] == " ]":
                            searched_time = datetime.strptime(
                                date_str[2:-2],
                                LoggingMixin._date_format_string_,
                            )
                        else:
                            msg_block.append(line)
                            continue
                        if last_searched is not None:
                            msg_block_ = "".join(msg_block)
                            msg_dict.setdefault(last_searched, []).append(msg_block_)
                        last_searched = searched_time
                        msg_block = [line]
                    if msg_block:
                        msg_dict.setdefault(last_searched, []).append(
                            "".join(msg_block)
                        )
            with open(tgt_file, "w") as f:
                f.write("".join(["".join(msg_dict[key]) for key in sorted(msg_dict)]))

    @classmethod
    def reset_logging(cls) -> None:
        cls._triggered_ = False
        cls._initialized_ = False
        cls._logging_path_ = None
        if cls._logger_ is not None:
            cls._release_handlers(cls._logger_)
        cls._logger_ = cls._verbose_level_ = None
        cls._timing_dict_, cls._time_cache_dict_ = {}, {}

    @classmethod
    def start_timer(cls, name):
        if name in cls._time_cache_dict_:
            print(
                f"'{name}' was already in time cache dict, "
                "this may cause by calling `start_timer` repeatedly"
            )
            return
        cls._time_cache_dict_[name] = time.time()

    @classmethod
    def end_timer(cls, name):
        start_time = cls._time_cache_dict_.pop(name, None)
        if start_time is None:
            print(
                f"'{name}' was not found in time cache dict, "
                "this may cause by not calling `start_timer` method"
            )
            return
        incrementer = cls._timing_dict_.setdefault(name, Incrementer())
        incrementer.update(time.time() - start_time)

    def log_timing(self):
        timing_str_list = ["=" * 138]
        for name in sorted(self._timing_dict_.keys()):
            incrementer = self._timing_dict_[name]
            timing_str_list.append(
                f"|   {name:<82s}   | "
                f"{fix_float_to_length(incrementer.mean, 10)} ± "
                f"{fix_float_to_length(incrementer.std, 10)} | "
                f"{incrementer.n_record:>12d} hits   |"
            )
            timing_str_list.append("-" * 138)
        self.log_block_msg(
            "\n".join(timing_str_list),
            title="timing",
            verbose_level=None,
            msg_level=logging.DEBUG,
        )
        return self