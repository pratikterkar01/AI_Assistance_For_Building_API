from pydantic import BaseModel, Field
from typing import Optional, List

class EvalRequest(BaseModel):
    questions: List[str]
    ground_truths: List[str]