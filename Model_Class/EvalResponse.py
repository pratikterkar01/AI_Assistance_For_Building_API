from pydantic import BaseModel, Field
from typing import Optional, List

class EvalResponse(BaseModel):
    faithfulness: float
    answer_relevancy: float
    context_recall: float