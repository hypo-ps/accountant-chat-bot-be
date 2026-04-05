"""
Pydantic models for API requests and responses.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# Request Models

class ChatMode(str, Enum):
    """Chat mode options."""
    DIRECT = "direct"      # Direct LLM, no RAG
    RAG_AUTO = "rag_auto"  # Agent decides when to use RAG (recommended)
    RAG_ALWAYS = "rag_always"  # Always retrieve before responding


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., description="User's message", min_length=1, max_length=10000)
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID")
    system_prompt: Optional[str] = Field(None, description="Override system prompt")
    use_tools: bool = Field(True, description="Whether to allow tool usage")
    stream: bool = Field(False, description="Whether to stream the response")
    mode: ChatMode = Field(
        ChatMode.RAG_AUTO,
        description="Chat mode: 'direct' (no RAG), 'rag_auto' (agent decides), 'rag_always' (always use RAG)"
    )


class ConversationCreate(BaseModel):
    """Request model for creating a conversation."""
    system_prompt: Optional[str] = Field(None, description="System prompt for the conversation")
    metadata: Optional[dict[str, Any]] = Field(None, description="Additional metadata")


class SystemPromptUpdate(BaseModel):
    """Request model for updating system prompt."""
    system_prompt: str = Field(..., description="New system prompt", min_length=1)


class DocumentIngestRequest(BaseModel):
    """Request model for document ingestion."""
    text: str = Field(..., description="Text content to ingest")
    source: str = Field("manual_input", description="Source identifier")
    metadata: Optional[dict[str, Any]] = Field(None, description="Additional metadata")


class RAGQueryRequest(BaseModel):
    """Request model for RAG query."""
    question: str = Field(..., description="Question to answer", min_length=1)
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    system_prompt: Optional[str] = Field(None, description="Custom system prompt")
    include_sources: bool = Field(True, description="Whether to include sources in response")


# Response Models

class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    content: str = Field(..., description="Assistant's response")
    conversation_id: str = Field(..., description="Conversation ID")
    tool_calls_made: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
    iterations: int = Field(1, description="Number of LLM iterations")


class ConversationResponse(BaseModel):
    """Response model for conversation info."""
    id: str
    message_count: int
    system_prompt: Optional[str]
    created_at: datetime
    updated_at: datetime


class ConversationListResponse(BaseModel):
    """Response model for conversation list."""
    conversations: list[dict[str, Any]]
    total: int


class MessageResponse(BaseModel):
    """Response model for a message."""
    role: str
    content: str
    name: Optional[str] = None


class ConversationDetailResponse(BaseModel):
    """Response model for conversation with messages."""
    id: str
    messages: list[MessageResponse]
    system_prompt: Optional[str]
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class DocumentIngestResponse(BaseModel):
    """Response model for document ingestion."""
    document_id: str
    source: str
    chunks_created: int


class RAGQueryResponse(BaseModel):
    """Response model for RAG query."""
    answer: str
    has_context: bool
    sources: Optional[list[dict[str, Any]]] = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    version: str
    llm_available: bool
    vectordb_available: bool


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str
    message: str
    details: Optional[dict[str, Any]] = None
