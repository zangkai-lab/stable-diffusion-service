import dill
import torch
import math
import uuid

from typing import Dict, List
from torch import Tensor

from models.model.model_condition import IConditionModel


@IConditionModel.register("multimodal/clip")
@IConditionModel.register("multimodal/clip.large")
@IConditionModel.register("multimodal/clip.chinese")
@IConditionModel.register("multimodal/clip.open_clip_ViT_H_14")
class CLIPTextConditionModel(IConditionModel):
    m: CLIP
    tokenizer: ICLIPTokenizer

    def __init__(self, m: CLIP):
        super().__init__(m)
        self.clip_skip = 0
        self.context_length = m.context_length
        tokenizer = "clip.chinese" if m.context_length == 512 else "clip"
        self.tokenizer = ITokenizer.make(tokenizer, {})
        self.comma_token = self.tokenizer.tokenizer.get_vocab()[",</w>"]
        self.comma_padding_backtrack = 20
        self._dumped_tokenizer = dill.dumps(self.tokenizer.tokenizer)
        self.dictionary: Dict[str, str] = {}
        self.customized: Dict[int, CustomToken] = {}

    @property
    def embeddings(self) -> Tensor:
        if self.m.token_embedding is None:
            raise ValueError("`token_embedding` is None")
        return self.m.token_embedding.weight.data

    def register_custom(self, embeddings: Dict[str, List[List[float]]]) -> None:
        existing = self.embeddings
        dtype = existing.dtype
        device = existing.device
        for name, embedding in embeddings.items():
            tensor = torch.asarray(embedding, dtype=dtype, device=device)
            embedding_dim = self.embeddings.shape[1]
            if tensor.shape[1] != embedding_dim:
                raise ValueError(
                    f"dimension of the custom embedding '{name}' is not correct "
                    f"(expected {embedding_dim}, got {tensor.shape[1]})"
                )
            tensor = tensor.view(-1, tensor.shape[-1])
            tags = []
            for i in range(tensor.shape[0]):
                i_tensor = tensor[[i]]
                tag = uuid.uuid4().hex
                self.tokenizer.tokenizer.add_tokens(tag)
                token_id = self.tokenizer.tokenizer.convert_tokens_to_ids(tag)
                self.customized[token_id] = CustomToken(name, tag, token_id, i_tensor)
                tags.append(tag)
            self.dictionary[name] = " ".join(tags)

    def clear_custom(self) -> None:
        self.dictionary = {}
        self.customized = {}
        # TODO : optimize this
        self.tokenizer.tokenizer = dill.loads(self._dumped_tokenizer)

    def get_line(self, text: str, clip_skip: int) -> Tensor:
        for k, v in self.dictionary.items():
            text = text.replace(k, v)
        parsed = parse_weights(text)
        parsed_texts = [pair[0] for pair in parsed]
        parsed_weights = [pair[1] for pair in parsed]
        tokenized = self.tokenizer.encode(
            parsed_texts,
            truncation=False,
            add_special_tokens=False,
        )["input_ids"]

        concat_ids: List[int] = []
        weights: List[float] = []
        id_end = self.tokenizer.eos_token_id
        last_comma = -1
        valid_context_length = self.context_length - 2
        for tokens, weight in zip(tokenized, parsed_weights):
            i = 0
            while i < len(tokens):
                token = tokens[i]
                if token == self.comma_token:
                    last_comma = len(concat_ids)
                elif (
                    self.comma_padding_backtrack != 0
                    and max(len(concat_ids), 1) % valid_context_length == 0
                    and last_comma != -1
                    and len(concat_ids) - last_comma <= self.comma_padding_backtrack
                ):
                    last_comma += 1
                    reloc_tokens = concat_ids[last_comma:]
                    reloc_mults = weights[last_comma:]

                    concat_ids = concat_ids[:last_comma]
                    length = len(concat_ids)

                    rem = (
                        int(math.ceil(length / valid_context_length))
                        * valid_context_length
                        - length
                    )
                    concat_ids += [id_end] * rem + reloc_tokens
                    weights = weights[:last_comma] + [1.0] * rem + reloc_mults

                concat_ids.append(token)
                weights.append(weight)
                i += 1

        zs = []
        is_first = True
        remained_ids = concat_ids
        remained_weights = weights
        while is_first or remained_ids:
            local_ids = remained_ids[:]
            local_weights = remained_weights[:]
            local_ids = [self.tokenizer.bos_token_id] + local_ids
            local_weights = [1.0] + local_weights
            # padding
            diff = self.context_length - len(local_ids)
            if diff > 0:
                local_ids += [self.tokenizer.eos_token_id] * diff
                local_weights += [1.0] * diff
                remained_ids = []
            else:
                remained_ids = local_ids[self.context_length - 1 :]
                remained_weights = local_weights[self.context_length - 1 :]
                local_ids = local_ids[: self.context_length - 1]
                local_weights = local_weights[: self.context_length - 1]
                local_ids.append(self.tokenizer.eos_token_id)
                local_weights.append(1.0)
            # encode
            to_torch = lambda l, dtype: torch.asarray(
                l,
                dtype=dtype,
                device=get_device(self.m),
            )
            inp = to_torch([local_ids], torch.int64)
            weights_tensor = to_torch(local_weights, torch.float32).view(1, -1, 1)
            with inject_embeddings(self):
                z = self.m.encode_text(
                    inp,
                    apply_pooling=False,
                    determinate=False,
                    clip_skip=clip_skip,
                )
            # weighting
            original_mean = z.mean()
            z *= weights_tensor
            new_mean = z.mean()
            z *= original_mean / new_mean
            zs.append(z)
            # set flag
            is_first = False

        if len(zs) == 1:
            return zs[0]
        return torch.cat(zs, dim=1)

    def forward(self, cond: List[str]) -> Tensor:
        lines = [self.get_line(text, self.clip_skip) for text in cond]
        return torch.cat(lines, dim=0)