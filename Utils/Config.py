from pydantic_settings import  BaseSettings
from pydantic import ConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    ## server setting
    HOST: str = "127.0.0.1"
    PORT: int = 8080
    DEBUG: bool = True

    # ChromaDB
    CHROMA_COLLECTION: str = "building_docs"
    CHROMA_PERSIST_DIR: str = "D:\\AiEnergyAssitanceForBuilding\\AiBuildingAssistanceApi\\ChromeDB"
    class ConfigDict:
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()