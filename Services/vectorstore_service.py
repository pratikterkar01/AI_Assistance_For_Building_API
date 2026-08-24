from langchain_chroma import Chroma
from Services.embedding_service import get_embedder
from Utils.Config import get_settings
from Utils.logger import get_logger
from typing import List
from langchain_core.documents import Document

logger = get_logger(__name__)
settings = get_settings()
_vectorstore = None

def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            collection_name=settings.CHROMA_COLLECTION,
            embedding_function=get_embedder(),
            persist_directory=settings.CHROMA_PERSIST_DIR,
        )
    return _vectorstore

def add_documents(chunks: List[Document]) -> int:
    vs = get_vectorstore()
    vs.add_documents(chunks)
    logger.info(f"Added {len(chunks)} chunks to ChromaDB")
    return len(chunks)

def get_retriever(search_type: str = "mmr"):
    vs = get_vectorstore()
    return vs.as_retriever(
        search_type=search_type,
        search_kwargs={"k": settings.TOP_K},
    )