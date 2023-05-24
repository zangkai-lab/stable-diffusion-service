from pydantic import BaseModel, Field
from fastapi import HTTPException


class HealthRequest(BaseModel):
    msg: str = Field(..., description="the message to ask")

    class Config:
        description = "Health Request"


class HealthResponse(BaseModel):
    status: str = Field(..., description="the message to answer")

    class Config:
        description = "Health Response"


async def health_check_handler(req: HealthRequest) -> HealthResponse:
    return HealthResponse(status="I am ok! still alive!")

