"""
Vector database module using Qdrant.

Qdrant is a high-performance vector database written in Rust.
Supports both self-hosted (Docker) and cloud deployments.
"""

from app.vectordb.base import BaseVectorDB, Document, SearchResult
from app.vectordb.qdrant import QdrantVectorDB
from app.vectordb.factory import create_vectordb

__all__ = [
    "BaseVectorDB",
    "Document",
    "SearchResult",
    "QdrantVectorDB",
    "create_vectordb",
]
