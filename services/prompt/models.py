from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

load_dotenv()


class GetPromptRequest(BaseModel):
    text: str
    need_translate: bool
    disable_audit: bool

    class Config:
        description = "GetPrompt Request"


class GetPromptResponse(BaseModel):
    text: str
    success: bool
    reason: str
    recommend_prompts: List[str] = []

    class Config:
        description = "GetPrompt Response"


