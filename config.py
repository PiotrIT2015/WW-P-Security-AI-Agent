"""
config.py

Central configuration for Secure RAG System.
Python 3.12+
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# Paths
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
LOG_DIR = DATA_DIR / "logs"
TEMP_DIR = DATA_DIR / "temp"

for directory in (
    DATA_DIR,
    DOCUMENTS_DIR,
    VECTORSTORE_DIR,
    LOG_DIR,
    TEMP_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Settings
# =============================================================================

class Settings(BaseSettings):
    """
    Secure RAG configuration.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # General
    # -------------------------------------------------------------------------

    APP_NAME: str = "Secure RAG"

    VERSION: str = "1.0.0"

    DEBUG: bool = False

    HOST: str = "127.0.0.1"

    PORT: int = 8000

    # -------------------------------------------------------------------------
    # Database
    # -------------------------------------------------------------------------

    DATABASE_URL: str = Field(
        default=f"sqlite:///{BASE_DIR / 'secure_rag.db'}"
    )

    # -------------------------------------------------------------------------
    # Authentication
    # -------------------------------------------------------------------------

    JWT_SECRET_KEY: str = Field(
    ...,
    min_length=32
    )

    JWT_ALGORITHM: str = "HS256"

    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # -------------------------------------------------------------------------
    # Encryption
    # -------------------------------------------------------------------------

    AES_KEY: str = Field(
    ...,
    min_length=44,
    max_length=44,
    description="Base64 encoded AES-256 key"
    )

    # -------------------------------------------------------------------------
    # Embeddings
    # -------------------------------------------------------------------------

    EMBEDDING_MODEL: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    CHUNK_SIZE: int = 500

    CHUNK_OVERLAP: int = 100

    TOP_K: int = 5

    # -------------------------------------------------------------------------
    # Ollama
    # -------------------------------------------------------------------------

    OLLAMA_URL: str = "http://localhost:11434"

    OLLAMA_MODEL: str = "phi3:latest"

    # -------------------------------------------------------------------------
    # FAISS
    # -------------------------------------------------------------------------

    FAISS_INDEX: str = str(
        VECTORSTORE_DIR / "index.faiss"
    )

    FAISS_METADATA: str = str(
        VECTORSTORE_DIR / "metadata.pkl"
    )

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    LOG_LEVEL: str = "INFO"

    LOG_FILE: str = str(
        LOG_DIR / "secure_rag.log"
    )

    # -------------------------------------------------------------------------
    # Security
    # -------------------------------------------------------------------------

    ENABLE_YARA: bool = True

    ENABLE_CLAMAV: bool = True

    ENABLE_PII_FILTER: bool = True

    ENABLE_PROMPT_GUARD: bool = True

    ENABLE_AUDIT: bool = True

    MAX_UPLOAD_SIZE_MB: int = 50

    ALLOWED_EXTENSIONS: list[str] = [
        ".pdf",
        ".txt",
        ".docx",
        ".md",
        ".html",
    ]

    # -------------------------------------------------------------------------
    # Rate limiting
    # -------------------------------------------------------------------------

    REQUESTS_PER_MINUTE: int = 100


settings = Settings()