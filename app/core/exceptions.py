"""
Custom exceptions for the chatbot application.

Provides structured error handling with clear error types.
"""

from typing import Any, Optional


class ChatbotException(Exception):
    """Base exception for all chatbot-related errors."""
    
    def __init__(
        self,
        message: str,
        details: Optional[dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.original_error = original_error
    
    def to_dict(self) -> dict[str, Any]:
        """Convert exception to a dictionary for API responses."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


class ConfigurationError(ChatbotException):
    """Raised when there's a configuration problem."""
    pass


class LLMError(ChatbotException):
    """Base exception for LLM-related errors."""
    pass


class LLMConnectionError(LLMError):
    """Raised when unable to connect to LLM provider."""
    pass


class LLMRateLimitError(LLMError):
    """Raised when LLM rate limit is exceeded."""
    pass


class LLMResponseError(LLMError):
    """Raised when LLM response is invalid or unexpected."""
    pass


class VectorDBError(ChatbotException):
    """Base exception for vector database errors."""
    pass


class VectorDBConnectionError(VectorDBError):
    """Raised when unable to connect to vector database."""
    pass


class DocumentProcessingError(ChatbotException):
    """Raised when document processing fails."""
    pass


class DocumentUploadError(DocumentProcessingError):
    """Raised when document upload fails."""
    pass


class DocumentParsingError(DocumentProcessingError):
    """Raised when document parsing fails."""
    pass


class RAGError(ChatbotException):
    """Base exception for RAG pipeline errors."""
    pass


class EmbeddingError(RAGError):
    """Raised when embedding generation fails."""
    pass


class RetrievalError(RAGError):
    """Raised when document retrieval fails."""
    pass


class AgentError(ChatbotException):
    """Base exception for agent-related errors."""
    pass


class ToolExecutionError(AgentError):
    """Raised when a tool execution fails."""
    pass


class AgentTimeoutError(AgentError):
    """Raised when agent execution times out."""
    pass


class ConversationError(ChatbotException):
    """Raised when there's an error with conversation management."""
    pass


class ValidationError(ChatbotException):
    """Raised when input validation fails."""
    pass
