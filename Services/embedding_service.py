from langchain_openai import OpenAIEmbeddings
from Utils.Config import get_settings
from langchain_huggingface import HuggingFaceEmbeddings
settings = get_settings()
_embedder = None

def get_embedder() -> OpenAIEmbeddings:
    global _embedder
    if _embedder is None:
        _embedder = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)
    return _embedder