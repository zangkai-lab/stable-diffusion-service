from enum import Enum


class PrecisionType(str, Enum):
    NO = "no"
    FP8 = "fp8"
    FP16 = "fp16"
    BF16 = "bf16"