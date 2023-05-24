from fastapi import FastAPI

from .health import HealthRequest
from .health import HealthResponse
from .health import health_check_handler

from .prompt import GetPromptRequest
from .prompt import GetPromptResponse
from .prompt import translate_text_baidu

app = FastAPI()


# 健康检查接口，必须有参数输入
@app.post("/health", responses={200: {"model": HealthResponse}})
async def health_check(req: HealthRequest) -> HealthResponse:
    return await health_check_handler(req)


# prompt接口，目前就是接百度翻译将中文prompt翻译成英文
@app.post("/get_prompt", responses={200: {"model": GetPromptResponse}})
def get_prompt(req: GetPromptRequest) -> GetPromptResponse:
    if req.need_translate:
        try:
            translated_text = translate_text_baidu(req.text)
            return GetPromptResponse(text=translated_text, success=True, reason="")
        except Exception as e:
            return GetPromptResponse(text="", success=False, reason=str(e))
    else:
        return GetPromptResponse(text=req.text, success=True, reason="")