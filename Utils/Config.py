from pydantic_settings import  BaseSettings
from functools import lru_cache
import os
class Settings(BaseSettings):
    ## server setting
    HOST: str = "127.0.0.1"
    PORT: int = 8080
    DEBUG: bool = True

    ## POSTGRES sql
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "pass123"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "ai_building_db"

    # ChromaDB
    CHROMA_COLLECTION: str = "building_docs"
    CHROMA_PERSIST_DIR: str = "D:\\AiEnergyAssitanceForBuilding\\AiBuildingAssistanceApi\\ChromeDB"
    CURRENT_DIR: str = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT: str = os.path.dirname(CURRENT_DIR)
    # 3. Create your DB path cleanly
    DB_ROOT: str = os.path.join(PROJECT_ROOT, "DB")
    UPLOAD_FILE_FOLDER:str = os.path.join(PROJECT_ROOT, "data\\raw")
    PARENTS_FILE: str = "parents.json"
    MANIFEST_FILE: str = "ingest_manifest.json"
    EMBEDDING_MODELS: dict = {
        "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    }

    CHUNKING_FOLDERS: dict = {
        "structure_aware": "StructureAwareChunking",
        "hierarchical": "HierarchicalChunking",
        "context_aware": "ContextuaTokenAwareChunking",
    }

    # --------------------------------------------------------------------------
    # CONFIG: change these to match how your 9 collections were written
    # --------------------------------------------------------------------------
    TEXT_FIELD: str = "text"  # payload key holding the child chunk text
    PARENT_ID_FIELD: str = "parent_id"  # payload key linking a child to its parent section
    PARENT_TEXT_FIELD: str = "parent_text"  # payload key with the full parent section (if stored)
    SOURCE_FIELD: str = "source"  # e.g. document name, used for citations
    SECTION_FIELD: str = "section"  # e.g. "6.5.1", used for citations
    VECTOR_NAME: str | None = None  # None for an unnamed vector, or e.g. "dense"
    DENSE:str = "dense"
    SPARSE:str = "bm25"
    K1:float = 1.5
    B:float = 0.75
    OVERLAP :float= 0.6  # set by --overlap
    TEXT_CANDIDATES:tuple = ("text", "page_content", "content", "chunk_text", "chunk")
    DENSE_TOP_K: int = 50
    BM25_TOP_K: int = 50
    RRF_K: int = 60
    RERANK_POOL: int = 30  # how many fused candidates the cross-encoder scores
    FINAL_TOP_K: int = 5
    RERANK_MAX_LEN: int = 100
    RERANKER_NAME: str = "BAAI/bge-reranker-v2-m3"
    TEXT_CANDIDATES: tuple = ("text", "page_content", "content", "chunk_text", "chunk")

    ABSTAIN_TEXT: str = "Not found in the documents."

    # ---- 1. Static prefix (system prompt + few-shot) ----
    # Kept identical on every request so the server can cache it (prefix caching).
    SYSTEM_PROMPT: str = f"""You are an assistant that answers questions about building codes \
        and energy standards using ONLY the numbered context clauses provided.

        Rules:
        1. Use only the information in the context. Do not use outside knowledge.
        2. After every claim, cite the supporting clause as [n], e.g. [1] or [2][3].
        3. Copy numbers, units and thresholds exactly as written in the context.
        4. If the context does not contain enough information to answer, reply exactly:
           "{ABSTAIN_TEXT}"
        5. Do not guess or invent clause numbers. Cite only the [n] labels shown.
        6. Be concise. Answer in at most 5 sentences.

        Examples (illustrative format only; the values are placeholders):

        Context:
        [1] (Doc A, Sec 1.1) The minimum value of X for category P is 10 units.
        [2] (Doc A, Sec 1.2) Category Q has no minimum requirement for X.
        Question: What is the minimum X for category P?
        Answer: The minimum X for category P is 10 units [1].

        Context:
        [1] (Doc B, Sec 2.1) Equipment Y shall be tested annually.
        Question: What is the maximum operating temperature of equipment Y?
        Answer: {ABSTAIN_TEXT}
        """

    # LLM
    provider: str = os.getenv("RAG_LLM_PROVIDER", "ollama")  # "ollama" | "openai"
    ollama_model: str = os.getenv("RAG_OLLAMA_MODEL", "qwen3:4b")
    ollama_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    openai_model: str = os.getenv("RAG_OPENAI_MODEL", "gpt-4o")  # set to the GPT-4 model you use
    num_ctx: int = int(os.getenv("RAG_NUM_CTX", "4096"))  # raise if you send bigger parents
    max_new_tokens: int = int(os.getenv("RAG_MAX_NEW_TOKENS", "400"))
    keep_alive: str = os.getenv("RAG_KEEP_ALIVE", "30m")  # keep the model loaded in VRAM
    # Retrieval -> augmentation
    retrieve_top_k: int = 5  # hits asked from the retriever (after rerank)
    max_parents: int = 3  # parent sections sent to the LLM
    parent_max_chars: int = 2400  # ~600 tokens per parent
    # Evidence gate. CrossEncoder gives 0-1 scores for bge-reranker-v2-m3 (sigmoid). TUNE on your golden set.
    min_rerank_score: float = float(os.getenv("RAG_MIN_RERANK", "0.05"))


    # ---- 2. Dynamic part (context + question) ----
    HUMAN_PROMPT: str = """Context:
        {context}

        Question: {question}
        Answer:"""
    ## Ingection
    PARSED_DIR:str = os.path.join(PROJECT_ROOT, "data\pdf_parsing")

    STAGE_MESSAGES :dict= {
        "parse": "The PDF could not be read. It may be corrupted or password-protected.",
        "chunk": "The document was read, but its text could not be split into sections.",
        "embed": "The document could not be converted for search. Please try again later.",
        "store": "The document could not be saved to the search database. Please try again later.",
    }

    class ConfigDict:
        env_file = ".env"

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

@lru_cache()
def get_settings():
    return Settings()