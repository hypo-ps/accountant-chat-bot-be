"""
Text splitting utilities for chunking documents.
"""

import re
from typing import Any, Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.documents.base import DocumentChunk, LoadedDocument

logger = get_logger(__name__)


class TextSplitter:
    """
    Split text into chunks with configurable size and overlap.
    
    Uses a recursive approach that tries to split on natural boundaries
    (paragraphs, sentences, words) before falling back to character-level splits.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[list[str]] = None,
    ) -> None:
        """
        Initialize the text splitter.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators to use for splitting (in order of preference)
        """
        self.chunk_size = chunk_size or settings.rag_chunk_size
        self.chunk_overlap = chunk_overlap or settings.rag_chunk_overlap
        self.separators = separators or [
            "\n\n",  # Paragraphs
            "\n",    # Lines
            ". ",    # Sentences
            "! ",
            "? ",
            "; ",
            ", ",    # Clauses
            " ",     # Words
            "",      # Characters
        ]
    
    def split_text(self, text: str) -> list[str]:
        """
        Split text into chunks.
        
        Args:
            text: The text to split
            
        Returns:
            List of text chunks
        """
        return self._split_text_recursive(text, self.separators)
    
    def _split_text_recursive(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using progressively smaller separators."""
        chunks: list[str] = []
        
        if not text.strip():
            return chunks
        
        # If text is small enough, return as is
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []
        
        # Try each separator
        for i, separator in enumerate(separators):
            if separator == "":
                # Character-level split as last resort
                chunks = self._split_by_characters(text)
                break
            
            if separator in text:
                splits = text.split(separator)
                
                # Merge small splits
                merged_chunks = self._merge_splits(splits, separator)
                
                # Recursively split chunks that are still too large
                for chunk in merged_chunks:
                    if len(chunk) > self.chunk_size and i + 1 < len(separators):
                        chunks.extend(self._split_text_recursive(chunk, separators[i + 1:]))
                    elif chunk.strip():
                        chunks.append(chunk.strip())
                break
        
        # Add overlap between chunks
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._add_overlap(chunks)
        
        return chunks
    
    def _merge_splits(self, splits: list[str], separator: str) -> list[str]:
        """Merge small splits together to approach chunk_size."""
        merged: list[str] = []
        current_chunk = ""
        
        for split in splits:
            test_chunk = current_chunk + separator + split if current_chunk else split
            
            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    merged.append(current_chunk)
                current_chunk = split
        
        if current_chunk:
            merged.append(current_chunk)
        
        return merged
    
    def _split_by_characters(self, text: str) -> list[str]:
        """Split text by character count."""
        chunks: list[str] = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at a word boundary
            if end < len(text):
                last_space = text.rfind(" ", start, end)
                if last_space > start:
                    end = last_space
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end
        
        return chunks
    
    def _add_overlap(self, chunks: list[str]) -> list[str]:
        """Add overlap between consecutive chunks."""
        if len(chunks) <= 1:
            return chunks
        
        overlapped: list[str] = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            curr_chunk = chunks[i]
            
            # Get the end of the previous chunk for overlap
            overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk
            
            # Find a good break point in the overlap
            space_idx = overlap_text.find(" ")
            if space_idx > 0:
                overlap_text = overlap_text[space_idx + 1:]
            
            overlapped.append(overlap_text + " " + curr_chunk)
        
        return overlapped
    
    def split_document(
        self,
        document: LoadedDocument,
        additional_metadata: Optional[dict[str, Any]] = None,
    ) -> list[DocumentChunk]:
        """
        Split a loaded document into chunks.
        
        Args:
            document: The document to split
            additional_metadata: Additional metadata to add to each chunk
            
        Returns:
            List of DocumentChunk objects
        """
        text_chunks = self.split_text(document.content)
        
        chunks: list[DocumentChunk] = []
        base_metadata = {**document.metadata, **(additional_metadata or {})}
        base_metadata["source"] = document.source
        
        for i, text in enumerate(text_chunks):
            chunk = DocumentChunk(
                content=text,
                metadata=base_metadata.copy(),
                document_id=document.document_id,
                chunk_index=i,
            )
            chunks.append(chunk)
        
        logger.debug(
            "Document split into chunks",
            document_id=document.document_id,
            num_chunks=len(chunks),
        )
        
        return chunks
