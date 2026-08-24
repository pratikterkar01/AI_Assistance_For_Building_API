import chromadb
from chromadb.config import Settings as ChromaSettings
from Utils.Config import get_settings
from Utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()
_client = None
_collection = None

def init_chromadb():
    global _client, _collection
    _client = chromadb.PersistentClient(
        path=settings.CHROMA_PERSIST_DIR,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    _collection = _client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(f"ChromaDB ready. Collection: {settings.CHROMA_COLLECTION}")
    return _collection

def get_collection():
    if _collection is None:
        return init_chromadb()
    return _collection

def get_client():
    if _client is None:
        init_chromadb()
    return _client


print(init_chromadb())