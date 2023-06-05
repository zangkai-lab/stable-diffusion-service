import operator

from functools import reduce
from typing import Iterable


def prod(iterable: Iterable) -> float:
    """Return cumulative production of an iterable."""

    return float(reduce(operator.mul, iterable, 1))


def truncate_string_to_length(string: str, length: int) -> str:
    """Truncate a string to make sure its length not exceeding a given length."""

    if len(string) <= length:
        return string
    half_length = int(0.5 * length) - 1
    head = string[:half_length]
    tail = string[-half_length:]
    return f"{head}{'.' * (length - 2 * half_length)}{tail}"