"""Centralized configuration. Relative paths resolve from the project root."""
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


class Settings(BaseModel):
    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b"
    openrouter_api_key: str = Field(default="", repr=False)
    openrouter_model: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    papers_directory: Path = Path("data/papers")
    vector_store_path: Path = Path("data/chroma")
    sessions_directory: Path = Path("data/sessions")
    log_directory: Path = Path("data/logs")
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)
    retrieval_top_k: int = Field(default=5, gt=0)

    @model_validator(mode="after")
    def validate_settings(self):
        if self.llm_provider not in {"ollama", "openrouter"}:
            raise ValueError("LLM_PROVIDER must be ollama or openrouter")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        for name in ("papers_directory", "vector_store_path", "sessions_directory", "log_directory"):
            path = getattr(self, name)
            if not path.is_absolute():
                setattr(self, name, ROOT / path)
        return self

    @classmethod
    def from_env(cls):
        fields = cls.model_fields
        values = {name: os.environ[name.upper()] for name in fields if name.upper() in os.environ}
        return cls.model_validate(values)


def get_settings() -> Settings:
    return Settings.from_env()
