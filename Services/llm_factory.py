from Utils.Config import get_settings
from Utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

def get_llm():
    if settings.LLM_MODE == "openai":
        from langchain_openai import ChatOpenAI
        logger.info("Using OpenAI GPT-4o")
        return ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0,
            openai_api_key=settings.OPENAI_API_KEY,
        )
    elif settings.LLM_MODE == "mistral":
        from langchain_community.llms import Ollama
        logger.info("Using Mistral via Ollama")
        return Ollama(
            model="mistral",
            base_url=settings.MISTRAL_BASE_URL,
        )
    else:
        raise ValueError(f"Unknown LLM_MODE: {settings.LLM_MODE}")