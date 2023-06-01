from fastapi import FastAPI, Response

from services.apis.prompt import GetPromptRequest
from services.apis.prompt import GetPromptResponse
from services.apis.prompt import translate_text_baidu

from services.apis.txt2img import txt2imgSDRequest
from services.apis.txt2img import txt2img_run
from services.apis.txt2img import txt2img_initialize

from services.client.client_http import HttpClient

app = FastAPI()
http_client = HttpClient()


# prompt接口，目前就是接百度翻译将中文prompt翻译成英文
@app.post("/get_prompt", responses={200: {"models": GetPromptResponse}})
def get_prompt(req: GetPromptRequest) -> GetPromptResponse:
    if req.need_translate:
        try:
            translated_text = translate_text_baidu(req.text)
            return GetPromptResponse(text=translated_text, success=True, reason="")
        except Exception as e:
            return GetPromptResponse(text="", success=False, reason=str(e))
    else:
        return GetPromptResponse(text=req.text, success=True, reason="")


@app.get("/txt2img", responses={200: {"models": Response}})
def txt2img(req: txt2imgSDRequest) -> Response:
    return await txt2img_run(req)


@app.on_event("startup")
async def startup() -> None:
    # 启动http客户端
    http_client.start()
    # 初始化txt2img
    txt2img_initialize()
    print("> Server is Ready!")


# 优雅的关闭服务器
@app.on_event("shutdown")
async def shutdown() -> None:
    await http_client.stop()
