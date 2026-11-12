from fastapi import APIRouter
from Model_Class.ChatResponse import  ChatResponse
from Model_Class.ChatRequest import ChatRequest
#from Services.rag_service import ask_question
from Utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    logger.info(f"Chat request: {request.question[:80]}")
    '''result = ask_question(request.question)
    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        session_id=request.session_id,
    )'''
    pass
