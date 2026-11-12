from pydantic import BaseModel, Field
from typing import Optional, List

class IngestResponse(BaseModel):
    status: str
    final_vector_db_result :List[dict]