"""
Document retriever for RAG pipeline.
"""

from typing import Any, Optional

from app.core.config import settings
from app.core.exceptions import RetrievalError
from app.core.logging import get_logger
from app.documents.base import DocumentChunk
from app.rag.embeddings import BaseEmbeddings
from app.vectordb.base import BaseVectorDB, Document, SearchResult

logger = get_logger(__name__)


class Retriever:
    """
    Retrieves relevant documents based on query similarity.
    
    Combines embeddings generation with vector database search.
    """
    
    def __init__(
        self,
        vectordb: BaseVectorDB,
        embeddings: BaseEmbeddings,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> None:
        """
        Initialize the retriever.
        
        Args:
            vectordb: Vector database instance
            embeddings: Embeddings instance
            top_k: Number of results to retrieve
            score_threshold: Minimum similarity score for results
        """
        self.vectordb = vectordb
        self.embeddings = embeddings
        self.top_k = top_k or settings.rag_top_k
        self.score_threshold = score_threshold
    
    async def add_documents(
        self,
        chunks: list[DocumentChunk],
    ) -> list[str]:
        """
        Add document chunks to the vector database.
        
        Args:
            chunks: List of document chunks to add
            
        Returns:
            List of document IDs that were added
        """
        try:
            # Generate embeddings for all chunks
            texts = [chunk.content for chunk in chunks]
            embeddings = await self.embeddings.embed_texts(texts)
            
            # Convert chunks to documents
            documents = [
                Document(
                    id=chunk.chunk_id,
                    content=chunk.content,
                    metadata={
                        **chunk.metadata,
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk_index,
                    },
                )
                for chunk in chunks
            ]
            
            # Add to vector database
            ids = await self.vectordb.add_documents(documents, embeddings)
            
            logger.info("Documents added to retriever", count=len(ids))
            return ids
            
        except Exception as e:
            raise RetrievalError(f"Failed to add documents: {e}", original_error=e)
    
    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: The search query
            top_k: Number of results (overrides default)
            filter_metadata: Optional metadata filter
            
        Returns:
            List of search results ordered by relevance
        """
        try:
            # Generate query embedding
            query_embedding = await self.embeddings.embed_text(query)
            
            # Search vector database
            results = await self.vectordb.search(
                query_embedding=query_embedding,
                top_k=top_k or self.top_k,
                filter_metadata=filter_metadata,
            )
            
            # Filter by score threshold
            filtered_results = [
                r for r in results
                if r.score >= self.score_threshold
            ]
            
            logger.debug(
                "Documents retrieved",
                query_length=len(query),
                results_count=len(filtered_results),
            )
            
            return filtered_results
            
        except Exception as e:
            raise RetrievalError(f"Retrieval failed: {e}", original_error=e)
    
    async def delete_documents(self, document_ids: list[str]) -> int:
        """
        Delete documents from the retriever.
        
        Args:
            document_ids: List of document IDs to delete
            
        Returns:
            Number of documents deleted
        """
        try:
            return await self.vectordb.delete_documents(document_ids)
        except Exception as e:
            raise RetrievalError(f"Failed to delete documents: {e}", original_error=e)
    
    async def get_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        max_context_length: int = 4000,
    ) -> str:
        """
        Get formatted context string for RAG.
        
        Args:
            query: The search query
            top_k: Number of results to retrieve
            max_context_length: Maximum length of context string
            
        Returns:
            Formatted context string for LLM
        """
        results = await self.retrieve(query, top_k=top_k)
        
        if not results:
            return ""
        
        context_parts: list[str] = []
        current_length = 0
        
        for i, result in enumerate(results):
            source = result.document.metadata.get("source", "Unknown")
            content = result.document.content
            
            part = f"[Source {i + 1}: {source}]\n{content}\n"
            
            if current_length + len(part) > max_context_length:
                break
            
            context_parts.append(part)
            current_length += len(part)
        
        return "\n".join(context_parts)
