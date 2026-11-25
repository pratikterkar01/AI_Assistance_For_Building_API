from fastapi import APIRouter
from RagServices.RagChainService import RagChainService
from Utils.logger import get_logger
from fastapi.responses import StreamingResponse
import json
router = APIRouter()
logger = get_logger(__name__)


@router.post("/ask_question", tags=["ask_question"])
async def ask_question(query: str):
    ## here we have copied then start the parsing to the docline doc and save the text file of it
    chain = RagChainService()

    def generate():
        try:
            for ev in chain.ask_query(query, mode="stream"):
                if ev["type"] == "token":
                    yield ev["text"]
                else:
                    yield "\n\nCitations: " + json.dumps(ev["result"]["citations"])
        except Exception as e:
            yield f"\n\n[error] {e}"
        finally:
            chain.client.close()  # runs after the answer is complete (or on error/disconnect)
    return StreamingResponse(generate(), media_type="text/plain")
