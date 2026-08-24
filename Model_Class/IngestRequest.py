from pydantic import BaseModel, Field
from typing import Optional, List

class IngestRequest(BaseModel):
    file_path: str
    doc_type: str = "ashrae"