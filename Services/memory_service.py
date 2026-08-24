from langchain.memory import ConversationBufferWindowMemory
from Utils.logger import get_logger

logger = get_logger(__name__)
_sessions: dict = {}

def get_memory(session_id: str) -> ConversationBufferWindowMemory:
    if session_id not in _sessions:
        _sessions[session_id] = ConversationBufferWindowMemory(
            k=5, return_messages=True, memory_key="chat_history"
        )
        logger.info(f"New memory session: {session_id}")
    return _sessions[session_id]

def clear_session(session_id: str):
    if session_id in _sessions:
        del _sessions[session_id]
        logger.info(f"Cleared session: {session_id}")