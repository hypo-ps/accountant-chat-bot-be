"""
Base document loader interface and data models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4


@dataclass
class DocumentChunk:
    """A chunk of a document after splitting."""
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: str = field(default_factory=lambda: str(uuid4()))
    document_id: Optional[str] = None
    chunk_index: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert chunk to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content": self.content,
            "metadata": self.metadata,
            "chunk_index": self.chunk_index,
        }


@dataclass
class LoadedDocument:
    """A loaded document before chunking."""
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    document_id: str = field(default_factory=lambda: str(uuid4()))
    source: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert document to dictionary."""
        return {
            "document_id": self.document_id,
            "content": self.content,
            "metadata": self.metadata,
            "source": self.source,
        }


class BaseDocumentLoader(ABC):
    """Abstract base class for document loaders."""
    
    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """List of supported file extensions."""
        pass
    
    def supports(self, file_path: Path) -> bool:
        """Check if this loader supports the given file."""
        return file_path.suffix.lower() in self.supported_extensions
    
    @abstractmethod
    async def load(self, file_path: Path) -> LoadedDocument:
        """
        Load a document from the given file path.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            LoadedDocument with content and metadata
        """
        pass
    
    @abstractmethod
    async def load_from_bytes(
        self,
        content: bytes,
        filename: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> LoadedDocument:
        """
        Load a document from bytes content.
        
        Args:
            content: The file content as bytes
            filename: Original filename
            metadata: Optional additional metadata
            
        Returns:
            LoadedDocument with content and metadata
        """
        pass


class DocumentLoaderRegistry:
    """Registry for document loaders."""
    
    def __init__(self) -> None:
        self._loaders: list[BaseDocumentLoader] = []
    
    def register(self, loader: BaseDocumentLoader) -> None:
        """Register a document loader."""
        self._loaders.append(loader)
    
    def get_loader(self, file_path: Path) -> Optional[BaseDocumentLoader]:
        """Get an appropriate loader for the given file."""
        for loader in self._loaders:
            if loader.supports(file_path):
                return loader
        return None
    
    def supported_extensions(self) -> list[str]:
        """Get all supported extensions."""
        extensions: list[str] = []
        for loader in self._loaders:
            extensions.extend(loader.supported_extensions)
        return list(set(extensions))


# Global registry instance
document_loader_registry = DocumentLoaderRegistry()
