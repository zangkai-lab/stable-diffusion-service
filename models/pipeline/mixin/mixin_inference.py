import torch
import numpy as np

from typing import Any, Callable, List, Optional, Type, Union

from models.data.data import IData
from models.pipeline.pipeline_base import Block
from models.pipeline.pipeline_inference import TInferPipeline
from models.pipeline.block.block_model import BuildModelBlock
from models.pipeline.block.block_inference import BuildInferenceBlock
from models.pipeline.block.block_serialize import SerializeDataBlock, SerializeModelBlock
from models.config.train_config import DLConfig
from models.model.constant import PREDICTIONS_KEY
from models.data.data_loader import IDataLoader
from models.utils.learn import softmax, sigmoid

from tools.utils.safe import safe_execute
from tools.utils.type import tensor_dict_type, np_dict_type
from tools.utils.icopy import shallow_copy_dict


class InferenceMixin:
    focuses: List[Type[Block]]
    is_built: bool

    data: Optional[IData]
    get_block: Callable[[Type[Block]], Any]
    try_get_block: Callable[[Type[Block]], Any]

    # optional callbacks

    def predict_callback(self, results: np_dict_type) -> np_dict_type:
        """changes can happen inplace"""
        return results

    # api

    @property
    def build_model(self) -> BuildModelBlock:
        return self.get_block(BuildModelBlock)

    @property
    def build_inference(self) -> BuildInferenceBlock:
        return self.get_block(BuildInferenceBlock)

    @property
    def serialize_data(self) -> SerializeDataBlock:
        return self.get_block(SerializeDataBlock)

    @property
    def serialize_model(self) -> Optional[SerializeModelBlock]:
        return self.try_get_block(SerializeModelBlock)

    @classmethod
    def build_with(  # type: ignore
        cls: Type[TInferPipeline],
        config: DLConfig,
        states: Optional[tensor_dict_type] = None,
        *,
        data: Optional[IData] = None,
    ) -> TInferPipeline:
        self = cls.init(config)
        # last focus will be the serialization block
        self.build(*[Block.make(b.__identifier__, {}) for b in cls.focuses])
        if states is not None:
            self.build_model.model.load_state_dict(states)
        self.serialize_model.verbose = False
        self.serialize_data.data = self.data = data
        self.is_built = True
        return self

    def to(  # type: ignore
        self: TInferPipeline,
        device: Union[int, str, torch.device],
    ) -> TInferPipeline:
        self.build_model.model.to(device)
        return self

    def predict(
        self,
        loader: IDataLoader,
        *,
        return_classes: bool = False,
        binary_threshold: float = 0.5,
        return_probabilities: bool = False,
        recover_labels: Optional[bool] = None,
        **kwargs: Any,
    ) -> np_dict_type:
        if not self.is_built:
            raise ValueError(
                f"`{self.__class__.__name__}` should be built beforehand, please use "
                "`DLPipelineSerializer.load_inference/evaluation` or `build_with` "
                "to get a built one!"
            )
        kw = shallow_copy_dict(kwargs)
        kw["loader"] = loader
        outputs = safe_execute(self.build_inference.inference.get_outputs, kw)
        results = outputs.forward_results
        # handle predict flags
        if return_classes and return_probabilities:
            raise ValueError(
                "`return_classes` & `return_probabilities`"
                "should not be True at the same time"
            )
        elif not return_classes and not return_probabilities:
            pass
        else:
            predictions = results[PREDICTIONS_KEY]
            if predictions.shape[1] > 2 and return_classes:
                results[PREDICTIONS_KEY] = predictions.argmax(1, keepdims=True)
            else:
                if predictions.shape[1] == 2:
                    probabilities = softmax(predictions)
                else:
                    pos = sigmoid(predictions)
                    probabilities = np.hstack([1.0 - pos, pos])
                if return_probabilities:
                    results[PREDICTIONS_KEY] = probabilities
                else:
                    classes = (probabilities[..., [1]] >= binary_threshold).astype(int)
                    results[PREDICTIONS_KEY] = classes
        # handle recover labels
        if recover_labels is None:
            recover_labels = self.data is not None
        if recover_labels:
            if self.data is None:
                msg = "`recover_labels` is set to `True` but `data` is not provided"
                raise ValueError(msg)
            y = results[PREDICTIONS_KEY]
            results[PREDICTIONS_KEY] = self.data.recover_labels(y)
        # optional callback
        results = self.predict_callback(results)
        # return
        return results