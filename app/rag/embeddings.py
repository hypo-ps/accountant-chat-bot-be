from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Optional

from openai import AsyncOpenAI, AsyncAzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import LLMProvider, settings
from app.core.exceptions import ConfigurationError, EmbeddingError
from app.core.logging import get_logger

logger = get_logger(__name__)


class BaseEmbeddings(ABC):
    """Abstract base class for embedding implementations."""
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass
    
    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: The text to embed
            
        Returns:
            Embedding vector as a list of floats
        """
        pass
    
    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass


class OpenAIEmbeddings(BaseEmbeddings):
    """OpenAI embeddings implementation."""
    
    # Embedding dimensions for different models
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key)
        self._dimension = self.MODEL_DIMENSIONS.get(model, 1536)
        logger.info("OpenAI Embeddings initialized", model=model)
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text using OpenAI."""
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embedding: {e}", original_error=e)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts using OpenAI."""
        if not texts:
            return []
        
        try:
            # OpenAI has a limit on batch size, so we chunk if necessary
            batch_size = 100
            all_embeddings: list[list[float]] = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            
            logger.debug("Generated embeddings", count=len(texts))
            return all_embeddings
            
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embeddings: {e}", original_error=e)


class AzureOpenAIEmbeddings(BaseEmbeddings):
    """Azure OpenAI embeddings implementation."""

    def __init__(
        self,
        azure_endpoint: str,
        api_key: str,
        deployment: str,
        api_version: str = "2024-12-01-preview",
        dimension: int = 1536,
        **kwargs: Any,
    ) -> None:
        self.deployment = deployment
        self._dimension = dimension
        self.client = AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        logger.info("Azure OpenAI Embeddings initialized", deployment=deployment)

    @property
    def dimension(self) -> int:
        return self._dimension

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding using Azure OpenAI."""
        try:
            response = await self.client.embeddings.create(
                model=self.deployment,
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            raise EmbeddingError(f"Azure embedding failed: {e}", original_error=e)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts using Azure OpenAI."""
        if not texts:
            return []

        try:
            batch_size = 100
            all_embeddings: list[list[float]] = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = await self.client.embeddings.create(
                    model=self.deployment,
                    input=batch,
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

            logger.debug("Generated Azure embeddings", count=len(texts))
            return all_embeddings

        except Exception as e:
            raise EmbeddingError(f"Azure embeddings failed: {e}", original_error=e)


def create_embeddings(**kwargs) -> BaseEmbeddings:
    """
    Create an embeddings instance based on LLM provider.

    Uses Azure OpenAI if LLM_PROVIDER is azure_openai, otherwise uses OpenAI.

    Args:
        **kwargs: Additional arguments

    Returns:
        An embeddings instance
    """
    provider = settings.llm_provider

    if provider == LLMProvider.AZURE_OPENAI:
        endpoint = kwargs.pop("azure_endpoint", None) or settings.azure_openai_endpoint
        api_key = kwargs.pop("api_key", None) or settings.azure_openai_api_key
        deployment = kwargs.pop("deployment", None) or settings.azure_openai_embedding_deployment

        if not endpoint or not api_key:
            raise ConfigurationError(
                "Azure OpenAI endpoint and API key required for embeddings",
                details={"provider": "azure_openai"},
            )
        if not deployment:
            raise ConfigurationError(
                "Azure OpenAI embedding deployment not configured",
                details={"config_key": "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"},
            )

        return AzureOpenAIEmbeddings(
            azure_endpoint=endpoint,
            api_key=api_key,
            deployment=deployment,
            api_version=kwargs.pop("api_version", None) or settings.azure_openai_api_version,
            **kwargs,
        )
    else:
        # Default to OpenAI
        api_key = kwargs.pop("api_key", None) or settings.openai_api_key
        if not api_key:
            raise ConfigurationError(
                "OpenAI API key required for embeddings",
                details={"provider": "openai"},
            )

        return OpenAIEmbeddings(
            api_key=api_key,
            model=kwargs.pop("model", None) or settings.embedding_model,
            **kwargs,
        )


@lru_cache
def get_embeddings() -> BaseEmbeddings:
    """Get a cached embeddings instance."""
    return create_embeddings()
