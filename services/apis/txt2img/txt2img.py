import os
from typing import Any
from fastapi import Response

from .interface import txt2imgSDRequest
from .init import init_sd

from services.pooling.pool_apis import api_pool
from tools.enum.apis import APIs
from tools.utils.to_type import get_bytes_from_diffusion


def txt2img_initialize() -> None:
    # apipool init
    api_pool.init(APIs.SD, init_sd)


async def txt2img_run(data: txt2imgSDRequest, **kwargs) -> Response:
    # load models
    m = get_sd_from(APIs.SD, data)
    # inference
    size = data.w, data.h
    img_arr = m.txt2img(
        data.text,
        size=size,
        max_wh=data.max_wh,
        **kwargs,
    ).numpy()[0]
    content = get_bytes_from_diffusion(img_arr)
    # apipool cleanup
    api_pool.cleanup(APIs.SD)
    # return response
    return Response(content=content, media_type="image/png")



