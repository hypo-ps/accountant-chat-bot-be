import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.routes import router
from app.core.config import settings, Environment
from app.core.logging import setup_logging, get_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger = get_logger(__name__)
    logger.info(
        "Starting application",
        app_name=settings.app_name,
        environment=settings.app_env.value,
        version=__version__,
    )
    
    # Setup directories
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    # Setup logging
    setup_logging(
        log_level=settings.log_level,
        json_format=settings.app_env == Environment.PRODUCTION,
    )
    
    app = FastAPI(
        title=settings.app_name,
        description="Production-ready agentic chatbot with LLM integration and RAG capabilities",
        version=__version__,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan,
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.debug else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routes
    app.include_router(router, prefix="/api/v1")
    
    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "name": settings.app_name,
            "version": __version__,
            "status": "running",
            "docs": "/docs" if settings.debug else "disabled",
        }
    
    return app


# Create application instance
app = create_app()


def main():
    """Run the application using uvicorn."""
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
