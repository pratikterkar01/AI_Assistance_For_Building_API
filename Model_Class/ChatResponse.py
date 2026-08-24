from pydantic import BaseModel, Field
from typing import Optional, List


class SourceChunk(BaseModel):
    content: str
    source: str
    page: Optional[int] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    session_id: str