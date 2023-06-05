from typing import OrderedDict as OrderedDictType
from collections import OrderedDict


class InjectDefaultsMixin:
    _defaults: OrderedDictType

    def __init__(self) -> None:
        self._defaults = OrderedDict()

    def process_defaults(self, _defaults: OrderedDictType) -> None:
        for k, v in self._defaults.items():
            _defaults[k] = v