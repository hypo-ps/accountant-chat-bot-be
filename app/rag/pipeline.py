"""
Complete RAG pipeline combining retrieval and generation.
"""

from pathlib import Path
from typing import Any, Optional

from app.core.config import settings
from app.core.exceptions import RAGError
from app.core.logging import get_logger
from app.documents.base import LoadedDocument, document_loader_registry
from app.documents.text_splitter import TextSplitter
from app.llm.base import BaseLLM, Message, MessageRole
from app.rag.embeddings import BaseEmbeddings
from app.rag.retriever import Retriever
from app.vectordb.base import BaseVectorDB

logger = get_logger(__name__)


class RAGPipeline:
    """
    Complete RAG pipeline for document-augmented generation.
    
    Combines document loading, chunking, embedding, retrieval, and generation.
    """
    
    RAG_SYSTEM_PROMPT_TEMPLATE = """You are a helpful AI assistant. Use the following context to answer the user's question. 
If you cannot find the answer in the context, say so clearly. Do not make up information.

Context:
{context}

---
Answer the user's question based on the context above."""
    
    def __init__(
        self,
        llm: BaseLLM,
        vectordb: BaseVectorDB,
        embeddings: BaseEmbeddings,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> None:
        """
        Initialize the RAG pipeline.
        
        Args:
            llm: LLM instance for generation
            vectordb: Vector database for storage
            embeddings: Embeddings for encoding
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            top_k: Number of documents to retrieve
        """
        self.llm = llm
        self.vectordb = vectordb
        self.embeddings = embeddings
        
        self.text_splitter = TextSplitter(
            chunk_size=chunk_size or settings.rag_chunk_size,
            chunk_overlap=chunk_overlap or settings.rag_chunk_overlap,
        )
        
        self.retriever = Retriever(
            vectordb=vectordb,
            embeddings=embeddings,
            top_k=top_k or settings.rag_top_k,
        )
        
        logger.info("RAG Pipeline initialized")
    
    async def ingest_document(
        self,
        file_path: Path,
        additional_metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Ingest a document into the RAG pipeline.
        
        Args:
            file_path: Path to the document
            additional_metadata: Extra metadata to attach
            
        Returns:
            Ingestion result with document ID and chunk count
        """
        try:
            # Get appropriate loader
            loader = document_loader_registry.get_loader(file_path)
            if not loader:
                raise RAGError(
                    f"Unsupported file type: {file_path.suffix}",
                    details={"supported": document_loader_registry.supported_extensions()},
                )
            
            # Load document
            document = await loader.load(file_path)
            
            # Split into chunks
            chunks = self.text_splitter.split_document(document, additional_metadata)
            
            # Add to retriever
            chunk_ids = await self.retriever.add_documents(chunks)
            
            result = {
                "document_id": document.document_id,
                "source": document.source,
                "chunks_created": len(chunk_ids),
                "chunk_ids": chunk_ids,
            }
            
            logger.info("Document ingested", **result)
            return result
            
        except Exception as e:
            raise RAGError(f"Document ingestion failed: {e}", original_error=e)
    
    async def ingest_text(
        self,
        text: str,
        source: str = "manual_input",
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Ingest raw text into the RAG pipeline.
        
        Args:
            text: The text content to ingest
            source: Source identifier
            metadata: Additional metadata
            
        Returns:
            Ingestion result
        """
        try:
            document = LoadedDocument(
                content=text,
                source=source,
                metadata=metadata or {},
            )
            
            chunks = self.text_splitter.split_document(document)
            chunk_ids = await self.retriever.add_documents(chunks)
            
            return {
                "document_id": document.document_id,
                "source": source,
                "chunks_created": len(chunk_ids),
            }
            
        except Exception as e:
            raise RAGError(f"Text ingestion failed: {e}", original_error=e)
    
    async def query(
        self,
        question: str,
        conversation_history: Optional[list[Message]] = None,
        system_prompt: Optional[str] = None,
        include_sources: bool = True,
    ) -> dict[str, Any]:
        """
        Query the RAG pipeline.
        
        Args:
            question: The user's question
            conversation_history: Previous messages
            system_prompt: Custom system prompt
            include_sources: Whether to include source information
            
        Returns:
            Response with answer and sources
        """
        try:
            # Retrieve relevant context
            context = await self.retriever.get_context(question)
            
            # Build system prompt with context
            rag_system_prompt = self.RAG_SYSTEM_PROMPT_TEMPLATE.format(context=context)
            if system_prompt:
                rag_system_prompt = f"{system_prompt}\n\n{rag_system_prompt}"
            
            # Build messages
            messages = list(conversation_history or [])
            messages.append(Message(role=MessageRole.USER, content=question))
            
            # Generate response
            response = await self.llm.generate(
                messages=messages,
                system_prompt=rag_system_prompt,
            )
            
            result: dict[str, Any] = {
                "answer": response.content,
                "has_context": bool(context),
            }
            
            if include_sources:
                # Get source information
                search_results = await self.retriever.retrieve(question)
                result["sources"] = [
                    {
                        "content": r.document.content[:200] + "...",
                        "score": r.score,
                        "metadata": r.document.metadata,
                    }
                    for r in search_results
                ]
            
            return result
            
        except Exception as e:
            raise RAGError(f"RAG query failed: {e}", original_error=e)
