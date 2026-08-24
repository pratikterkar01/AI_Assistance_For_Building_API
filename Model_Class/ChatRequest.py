from pydantic import BaseModel, Field
from typing import Optional, List

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    session_id: Optional[str] = "default"
    use_memory: bool = True