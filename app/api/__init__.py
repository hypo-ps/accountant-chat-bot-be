"""API module with FastAPI routes and dependencies."""

from app.api.routes import router
from app.api.dependencies import get_agent, get_rag_pipeline

__all__ = ["router", "get_agent", "get_rag_pipeline"]
