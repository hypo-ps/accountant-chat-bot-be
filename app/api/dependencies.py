"""
FastAPI dependencies for dependency injection.
"""

from functools import lru_cache
from typing import Optional

from app.agent.agent import Agent
from app.agent.rag_agent import RAGAgent
from app.agent.conversation import ConversationManager
from app.agent.tools import ToolRegistry, tool_registry
from app.core.config import settings
from app.core.logging import get_logger
from app.llm.factory import create_llm
from app.llm.base import BaseLLM
from app.rag.embeddings import create_embeddings, BaseEmbeddings
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import Retriever
from app.vectordb.factory import create_vectordb
from app.vectordb.qdrant import QdrantVectorDB

logger = get_logger(__name__)

# Cached instances
_llm_instance: Optional[BaseLLM] = None
_embeddings_instance: Optional[BaseEmbeddings] = None
_vectordb_instance: Optional[QdrantVectorDB] = None
_agent_instance: Optional[Agent] = None
_rag_agent_instance: Optional[RAGAgent] = None
_rag_pipeline_instance: Optional[RAGPipeline] = None
_retriever_instance: Optional[Retriever] = None
_conversation_manager: Optional[ConversationManager] = None


async def get_llm() -> BaseLLM:
    """Get or create LLM instance."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = create_llm()
    return _llm_instance


async def get_embeddings() -> BaseEmbeddings:
    """Get or create embeddings instance."""
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = create_embeddings()
    return _embeddings_instance


async def get_vectordb() -> QdrantVectorDB:
    """Get or create Qdrant vector database instance."""
    global _vectordb_instance
    if _vectordb_instance is None:
        _vectordb_instance = await create_vectordb()
    return _vectordb_instance


def get_conversation_manager() -> ConversationManager:
    """Get or create conversation manager."""
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = ConversationManager(
            default_system_prompt=settings.default_system_prompt
        )
    return _conversation_manager


async def get_agent() -> Agent:
    """Get or create agent instance."""
    global _agent_instance
    if _agent_instance is None:
        llm = await get_llm()
        conversation_manager = get_conversation_manager()
        _agent_instance = Agent(
            llm=llm,
            tool_registry=tool_registry,
            conversation_manager=conversation_manager,
            max_iterations=settings.agent_max_iterations,
            timeout_seconds=settings.agent_timeout_seconds,
        )
    return _agent_instance


async def get_retriever() -> Retriever:
    """Get or create retriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        embeddings = await get_embeddings()
        vectordb = await get_vectordb()
        _retriever_instance = Retriever(
            vectordb=vectordb,
            embeddings=embeddings,
            top_k=settings.rag_top_k,
        )
    return _retriever_instance


async def get_rag_pipeline() -> RAGPipeline:
    """Get or create RAG pipeline instance."""
    global _rag_pipeline_instance
    if _rag_pipeline_instance is None:
        llm = await get_llm()
        embeddings = await get_embeddings()
        vectordb = await get_vectordb()
        _rag_pipeline_instance = RAGPipeline(
            llm=llm,
            vectordb=vectordb,
            embeddings=embeddings,
        )
    return _rag_pipeline_instance


async def get_rag_agent() -> RAGAgent:
    """Get or create RAG-enabled agent instance."""
    global _rag_agent_instance
    if _rag_agent_instance is None:
        llm = await get_llm()
        retriever = await get_retriever()
        _rag_agent_instance = RAGAgent(
            llm=llm,
            retriever=retriever,
            max_iterations=settings.agent_max_iterations,
            timeout_seconds=settings.agent_timeout_seconds,
        )
    return _rag_agent_instance


async def reset_all() -> None:
    """Reset all cached instances (useful for testing)."""
    global _llm_instance, _embeddings_instance, _vectordb_instance
    global _agent_instance, _rag_agent_instance, _rag_pipeline_instance
    global _retriever_instance, _conversation_manager

    _llm_instance = None
    _embeddings_instance = None
    _vectordb_instance = None
    _agent_instance = None
    _rag_agent_instance = None
    _rag_pipeline_instance = None
    _retriever_instance = None
    _conversation_manager = None

    logger.info("All dependencies reset")
