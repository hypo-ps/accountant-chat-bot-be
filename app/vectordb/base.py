"""
Base vector database interface and data models.

Provides abstract base class for vector database implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4


@dataclass
class Document:
    """A document to store in the vector database."""
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4()))
    embedding: Optional[list[float]] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert document to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
        }


@dataclass
class SearchResult:
    """A search result from the vector database."""
    document: Document
    score: float
    
    def to_dict(self) -> dict[str, Any]:
        """Convert search result to dictionary."""
        return {
            "document": self.document.to_dict(),
            "score": self.score,
        }


class BaseVectorDB(ABC):
    """Abstract base class for vector database implementations."""
    
    def __init__(self, collection_name: str, **kwargs: Any) -> None:
        self.collection_name = collection_name
        self.extra_config = kwargs
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector database connection and collection."""
        pass
    
    @abstractmethod
    async def add_documents(
        self,
        documents: list[Document],
        embeddings: Optional[list[list[float]]] = None,
    ) -> list[str]:
        """
        Add documents to the vector database.
        
        Args:
            documents: List of documents to add
            embeddings: Optional pre-computed embeddings
            
        Returns:
            List of document IDs that were added
        """
        pass
    
    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: The query embedding vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of search results ordered by similarity
        """
        pass
    
    @abstractmethod
    async def delete_documents(self, document_ids: list[str]) -> int:
        """
        Delete documents by their IDs.
        
        Args:
            document_ids: List of document IDs to delete
            
        Returns:
            Number of documents deleted
        """
        pass
    
    @abstractmethod
    async def get_document(self, document_id: str) -> Optional[Document]:
        """
        Get a document by its ID.
        
        Args:
            document_id: The document ID
            
        Returns:
            The document if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def update_document(
        self,
        document_id: str,
        content: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        embedding: Optional[list[float]] = None,
    ) -> bool:
        """
        Update a document.
        
        Args:
            document_id: The document ID
            content: New content (optional)
            metadata: New metadata (optional)
            embedding: New embedding (optional)
            
        Returns:
            True if updated, False if not found
        """
        pass
    
    @abstractmethod
    async def count(self) -> int:
        """Get the total number of documents in the collection."""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all documents from the collection."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the vector database is accessible."""
        pass
