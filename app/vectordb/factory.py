"""
Vector database factory for Qdrant.

Qdrant is a high-performance vector database written in Rust.
Supports both self-hosted (Docker) and cloud deployments.
"""

from typing import Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.vectordb.base import BaseVectorDB
from app.vectordb.qdrant import QdrantVectorDB

logger = get_logger(__name__)


async def create_vectordb(**kwargs) -> BaseVectorDB:
    """
    Create a Qdrant vector database instance.

    Args:
        **kwargs: Additional arguments to pass to QdrantVectorDB

    Returns:
        An initialized Qdrant vector database instance
    """
    logger.info("Creating Qdrant vector database instance")

    vectordb = QdrantVectorDB(
        collection_name=kwargs.pop("collection_name", None) or settings.qdrant_collection_name,
        host=kwargs.pop("host", None) or settings.qdrant_host,
        port=kwargs.pop("port", None) or settings.qdrant_port,
        url=kwargs.pop("url", None) or settings.qdrant_url,
        api_key=kwargs.pop("api_key", None) or settings.qdrant_api_key,
        vector_size=kwargs.pop("vector_size", None) or settings.embedding_dimension,
        **kwargs,
    )
    await vectordb.initialize()
    return vectordb


# Cache for initialized vector database
_vectordb_instance: Optional[BaseVectorDB] = None


async def get_vectordb() -> BaseVectorDB:
    """
    Get a cached Qdrant vector database instance.

    Returns:
        A cached, initialized Qdrant instance
    """
    global _vectordb_instance

    if _vectordb_instance is None:
        _vectordb_instance = await create_vectordb()

    return _vectordb_instance


async def reset_vectordb() -> None:
    """Reset the cached vector database instance."""
    global _vectordb_instance
    _vectordb_instance = None
