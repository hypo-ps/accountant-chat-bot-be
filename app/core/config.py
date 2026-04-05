"""
Application configuration using Pydantic Settings.

Supports loading from environment variables and .env files.
"""

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Application environment."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""
    OPENAI = "openai"





class Settings(BaseSettings):
    """Application settings with validation."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Application Settings
    app_name: str = Field(default="accountant-chatbot", description="Application name")
    app_env: Environment = Field(default=Environment.DEVELOPMENT, description="Environment")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Server Settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    
    # LLM Provider Configuration
    llm_provider: LLMProvider = Field(default=LLMProvider.OPENAI, description="LLM provider")
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(default="gpt-4-turbo-preview", description="OpenAI model")
    openai_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    openai_max_tokens: int = Field(default=4096, ge=1)
    
    # Anthropic Configuration
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    anthropic_model: str = Field(default="claude-3-opus-20240229", description="Anthropic model")
    anthropic_temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    anthropic_max_tokens: int = Field(default=4096, ge=1)
    
    # Embedding Configuration
    embedding_model: str = Field(default="text-embedding-3-small", description="Embedding model")
    
    # Qdrant Vector Database Configuration
    qdrant_host: str = Field(default="localhost", description="Qdrant host")
    qdrant_port: int = Field(default=6333, description="Qdrant port")
    qdrant_url: Optional[str] = Field(default=None, description="Qdrant Cloud URL (overrides host/port)")
    qdrant_api_key: Optional[str] = Field(default=None, description="Qdrant API key (for cloud)")
    qdrant_collection_name: str = Field(default="documents", description="Qdrant collection name")
    embedding_dimension: int = Field(default=1536, description="Embedding vector dimension")
    
    # RAG Configuration
    rag_chunk_size: int = Field(default=1000, ge=100)
    rag_chunk_overlap: int = Field(default=200, ge=0)
    rag_top_k: int = Field(default=5, ge=1)
    
    # Document Upload Configuration
    upload_dir: Path = Field(default=Path("./data/uploads"))
    max_upload_size_mb: int = Field(default=50, ge=1)
    allowed_extensions: str = Field(default=".pdf,.txt,.docx,.md")
    
    # System Prompt (loaded from prompts.py)
    default_system_prompt: str = Field(default="")
    
    # Agent Configuration
    agent_max_iterations: int = Field(default=10, ge=1)
    agent_timeout_seconds: int = Field(default=120, ge=10)
    
    @property
    def allowed_extensions_list(self) -> list[str]:
        """Get allowed extensions as a list."""
        return [ext.strip() for ext in self.allowed_extensions.split(",")]
    
    @property
    def max_upload_size_bytes(self) -> int:
        """Get max upload size in bytes."""
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
