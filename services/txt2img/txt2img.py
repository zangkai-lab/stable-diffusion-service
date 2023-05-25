from typing import Any
from fastapi import Response

from .models import txt2imgSDRequest
from .utils import get_bytes_from_diffusion
from modules.apis import APIs


async def txt2imgSDRun(data: txt2imgSDRequest, **kwargs) -> Response:
    # load model
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



