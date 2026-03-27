"""
S4 Research Intelligence — Configuration
Central settings with environment variable overrides.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="S4RI_", extra="ignore")

    # --- Paths ---
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = project_root / "data"
    raw_dir: Path = data_dir / "raw"
    processed_dir: Path = data_dir / "processed"
    vector_dir: Path = data_dir / "vectors"

    # --- Embedding model ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"  # "cuda" if GPU available

    # --- LLM ---
    llm_provider: str = "ollama"  # "ollama" | "openai" | "aws_bedrock"
    llm_model: str = "mistral:7b-instruct-v0.3-q5_K_M"
    llm_base_url: str = "http://localhost:11434"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048
    llm_num_gpu: int = -1  # -1 = auto, 0 = CPU only
    llm_timeout: int = 120  # seconds

    # --- Vector store ---
    chroma_collection: str = "s4_research"
    chroma_persist_dir: str = str(vector_dir)

    # --- Retrieval ---
    retrieval_top_k: int = 8
    retrieval_score_threshold: float = 0.25
    rerank_enabled: bool = True
    rerank_top_n: int = 5
    hybrid_search_enabled: bool = True
    hybrid_semantic_weight: float = 0.7
    hybrid_keyword_weight: float = 0.3

    # --- Chunking ---
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # --- Source metadata ---
    source_types: list[str] = [
        "interview_transcript",
        "government_document",
        "archival_reference",
        "news_article",
        "production_note",
        "eyewitness_account",
        "scientific_paper",
        "book_excerpt",
    ]

    source_reliability_weights: dict[str, float] = {
        "government_document": 0.95,
        "scientific_paper": 0.90,
        "eyewitness_account": 0.75,
        "interview_transcript": 0.70,
        "archival_reference": 0.85,
        "news_article": 0.60,
        "book_excerpt": 0.65,
        "production_note": 0.50,
    }

    # --- API ---
    max_upload_size_mb: int = 50
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8501"]

    # --- Logging ---
    log_level: str = "INFO"


settings = Settings()
