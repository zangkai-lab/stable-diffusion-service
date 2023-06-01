from pydantic import BaseModel, Field
from typing import Optional

from modules.sampler import Samplers


class PromptInfo(BaseModel):
    lang: Optional[str]
    original: Optional[str]
    originalNegative: Optional[str]
    style: Optional[str]
    artist: Optional[str]


class txt2imgSDRequest(BaseModel):
    # models
    is_anime: bool = Field(
        True,
        description="anime or not"
    )
    model: str = Field(
        "anime",
        description="models name"
    )
    # seed
    use_seed: bool = Field(
        True,
        description="use seed or not"
    )
    seed: int = Field(
        -1,
        ge=-1,
        lt=2 ** 32,
        description="seed"
    )
    # image
    w: int = Field(
        512,
        le=1024,
        description="width"
    )
    h: int = Field(
        512,
        le=1024,
        description="height"
    )
    max_wh: int = Field(
        1024,
        description="Max width or height of the output image."
    )
    # prompt
    promptInfo: PromptInfo = Field(
        None,
        description="origin prompt info"
    )
    text: str = Field(
        ...,
        description="prompt text"
    )
    negative_prompt: str = Field(
        "",
        description="negative prompt text"
    )
    # noise
    use_circular: bool = Field(
        False,
        description="use circular or not"
    )
    sampler: Samplers = Field(
        Samplers.K_EULER,
        description="sampler"
    )
    num_steps: int = Field(
        20,
        ge=10,
        le=100,
        description="number of steps"
    )
    guidance_scale: int = Field(
        7.5,
        description="guidance scale"
    )
    # others
    timestamp: int = Field(
        0,
        description="timestamp"
    )

    class Config:
        description = "txt2imgSD Request"
