from pydantic import BaseModel, Field
from typing import Optional, List

class IngestResponse(BaseModel):
    status: str
    chunks_added: int
    collection: str