from fastapi import FastAPI

from .health import HealthRequest
from .health import HealthResponse
from .health import health_check_handler

app = FastAPI()


# 健康检查接口，必须有参数输入
@app.post("/health", responses={200: {"model": HealthResponse}})
async def health_check(req: HealthRequest) -> HealthResponse:
    return await health_check_handler(req)
