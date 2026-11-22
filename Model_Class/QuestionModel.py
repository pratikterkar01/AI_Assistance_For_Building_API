from pydantic import BaseModel
from typing import Optional

class QuestionModel(BaseModel):
    id: int
    category: Optional[str] = None
    question: str
    answer: Optional[str] = None
    sources: Optional[str] = None
    section: Optional[str] = None
    query_type: Optional[str] = None
    difficulty: Optional[str] = None
    gold_pages: Optional[str] = None
    verification_status: Optional[str] = None