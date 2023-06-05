from typing import TypeVar

TInferPipeline = TypeVar("TInferPipeline", bound="DLInferencePipeline", covariant=True)