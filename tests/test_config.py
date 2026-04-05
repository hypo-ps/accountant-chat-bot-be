"""Tests for configuration module."""

import pytest
from app.core.config import Settings, Environment, LLMProvider, VectorDBProvider


def test_default_settings():
    """Test default settings values."""
    settings = Settings()
    
    assert settings.app_name == "accountant-chatbot"
    assert settings.app_env == Environment.DEVELOPMENT
    assert settings.debug is False
    assert settings.llm_provider == LLMProvider.OPENAI
    assert settings.vector_db_provider == VectorDBProvider.CHROMA


def test_allowed_extensions_list():
    """Test parsing of allowed extensions."""
    settings = Settings(allowed_extensions=".pdf,.txt,.docx")
    
    assert settings.allowed_extensions_list == [".pdf", ".txt", ".docx"]


def test_max_upload_size_bytes():
    """Test conversion of upload size to bytes."""
    settings = Settings(max_upload_size_mb=10)
    
    assert settings.max_upload_size_bytes == 10 * 1024 * 1024


def test_environment_enum():
    """Test environment enum values."""
    assert Environment.DEVELOPMENT.value == "development"
    assert Environment.STAGING.value == "staging"
    assert Environment.PRODUCTION.value == "production"


def test_llm_provider_enum():
    """Test LLM provider enum values."""
    assert LLMProvider.OPENAI.value == "openai"
    assert LLMProvider.ANTHROPIC.value == "anthropic"


def test_vector_db_provider_enum():
    """Test vector DB provider enum values."""
    assert VectorDBProvider.CHROMA.value == "chroma"
    assert VectorDBProvider.PINECONE.value == "pinecone"
    assert VectorDBProvider.QDRANT.value == "qdrant"
