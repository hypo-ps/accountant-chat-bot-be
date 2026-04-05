"""
FastAPI routes for the chatbot API.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app import __version__
from app.api.dependencies import (
    get_agent,
    get_rag_agent,
    get_rag_pipeline,
    get_conversation_manager,
    get_llm,
    get_vectordb,
)
from app.api.models import (
    ChatMode,
    ChatRequest,
    ChatResponse,
    ConversationCreate,
    ConversationDetailResponse,
    ConversationListResponse,
    ConversationResponse,
    DocumentIngestRequest,
    DocumentIngestResponse,
    ErrorResponse,
    HealthResponse,
    MessageResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    SystemPromptUpdate,
)
from app.agent.agent import Agent
from app.agent.rag_agent import RAGAgent
from app.agent.conversation import ConversationManager
from app.core.exceptions import ChatbotException
from app.core.logging import get_logger
from app.rag.pipeline import RAGPipeline

logger = get_logger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Check the health status of the chatbot service."""
    llm_available = False
    vectordb_available = False
    
    try:
        llm = await get_llm()
        llm_available = await llm.health_check()
    except Exception:
        pass
    
    try:
        vectordb = await get_vectordb()
        vectordb_available = await vectordb.health_check()
    except Exception:
        pass
    
    return HealthResponse(
        status="healthy" if llm_available else "degraded",
        version=__version__,
        llm_available=llm_available,
        vectordb_available=vectordb_available,
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={500: {"model": ErrorResponse}},
    tags=["Chat"],
)
async def chat(
    request: ChatRequest,
    agent: Agent = Depends(get_agent),
    rag_agent: RAGAgent = Depends(get_rag_agent),
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> ChatResponse:
    """
    Send a message to the chatbot and receive a response.

    **Chat Modes:**
    - `direct`: Uses LLM directly without RAG (fastest, for general questions)
    - `rag_auto`: Agent intelligently decides when to search knowledge base (recommended)
    - `rag_always`: Always retrieves from knowledge base before responding

    **Other Features:**
    - Conversation history (provide conversation_id)
    - Custom system prompts
    - Tool calling
    - Streaming (set stream=true, only works with 'direct' mode)
    """
    try:
        # Handle streaming (only supported in direct mode)
        if request.stream:
            if request.mode != ChatMode.DIRECT:
                raise HTTPException(
                    status_code=400,
                    detail="Streaming only supported in 'direct' mode"
                )

            async def generate():
                async for chunk in agent.chat_stream(
                    message=request.message,
                    conversation_id=request.conversation_id,
                    system_prompt=request.system_prompt,
                ):
                    yield chunk

            return StreamingResponse(generate(), media_type="text/plain")

        # Handle different modes
        if request.mode == ChatMode.DIRECT:
            # Direct LLM without RAG
            response = await agent.chat(
                message=request.message,
                conversation_id=request.conversation_id,
                system_prompt=request.system_prompt,
                use_tools=request.use_tools,
            )

        elif request.mode == ChatMode.RAG_AUTO:
            # Agent decides when to use RAG (has search_knowledge_base tool)
            response = await rag_agent.chat(
                message=request.message,
                conversation_id=request.conversation_id,
                system_prompt=request.system_prompt,
                use_tools=True,  # Must be true for RAG tools
            )

        elif request.mode == ChatMode.RAG_ALWAYS:
            # Always retrieve context first
            result = await rag_pipeline.query(
                question=request.message,
                system_prompt=request.system_prompt,
                include_sources=True,
            )
            return ChatResponse(
                content=result["answer"],
                conversation_id="",  # RAG pipeline doesn't track conversations
                tool_calls_made=[{
                    "tool": "search_knowledge_base",
                    "arguments": {"query": request.message},
                    "result": {"sources_found": len(result.get("sources", []))}
                }],
                usage={},
                iterations=1,
            )

        return ChatResponse(
            content=response.content,
            conversation_id=response.conversation_id,
            tool_calls_made=response.tool_calls_made,
            usage=response.usage,
            iterations=response.iterations,
        )

    except ChatbotException as e:
        logger.error("Chat error", error=str(e), details=e.details)
        raise HTTPException(status_code=500, detail=e.to_dict())
    except Exception as e:
        logger.exception("Unexpected chat error")
        raise HTTPException(status_code=500, detail={"error": "InternalError", "message": str(e)})


# Conversation Management Routes

@router.get("/conversations", response_model=ConversationListResponse, tags=["Conversations"])
async def list_conversations(
    conversation_manager: ConversationManager = Depends(get_conversation_manager),
) -> ConversationListResponse:
    """List all conversations."""
    conversations = conversation_manager.list_conversations()
    return ConversationListResponse(conversations=conversations, total=len(conversations))


@router.post("/conversations", response_model=ConversationResponse, tags=["Conversations"])
async def create_conversation(
    request: ConversationCreate,
    conversation_manager: ConversationManager = Depends(get_conversation_manager),
) -> ConversationResponse:
    """Create a new conversation."""
    conversation = conversation_manager.create(
        system_prompt=request.system_prompt,
        metadata=request.metadata,
    )
    return ConversationResponse(
        id=conversation.id,
        message_count=len(conversation.messages),
        system_prompt=conversation.system_prompt,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationDetailResponse, tags=["Conversations"])
async def get_conversation(
    conversation_id: str,
    conversation_manager: ConversationManager = Depends(get_conversation_manager),
) -> ConversationDetailResponse:
    """Get a conversation with its messages."""
    conversation = conversation_manager.get(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return ConversationDetailResponse(
        id=conversation.id,
        messages=[
            MessageResponse(role=m.role.value, content=m.content, name=m.name)
            for m in conversation.messages
        ],
        system_prompt=conversation.system_prompt,
        metadata=conversation.metadata,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


@router.delete("/conversations/{conversation_id}", tags=["Conversations"])
async def delete_conversation(
    conversation_id: str,
    conversation_manager: ConversationManager = Depends(get_conversation_manager),
) -> dict:
    """Delete a conversation."""
    deleted = conversation_manager.delete(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"deleted": True, "conversation_id": conversation_id}


@router.put("/conversations/{conversation_id}/system-prompt", tags=["Conversations"])
async def update_system_prompt(
    conversation_id: str,
    request: SystemPromptUpdate,
    conversation_manager: ConversationManager = Depends(get_conversation_manager),
) -> dict:
    """Update the system prompt for a conversation."""
    updated = conversation_manager.update_system_prompt(conversation_id, request.system_prompt)
    if not updated:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"updated": True, "conversation_id": conversation_id}


# RAG Routes

@router.post(
    "/rag/ingest",
    response_model=DocumentIngestResponse,
    tags=["RAG"],
)
async def ingest_document(
    request: DocumentIngestRequest,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> DocumentIngestResponse:
    """Ingest text content into the RAG pipeline."""
    try:
        result = await rag_pipeline.ingest_text(
            text=request.text,
            source=request.source,
            metadata=request.metadata,
        )
        return DocumentIngestResponse(
            document_id=result["document_id"],
            source=result["source"],
            chunks_created=result["chunks_created"],
        )
    except ChatbotException as e:
        logger.error("Ingestion error", error=str(e))
        raise HTTPException(status_code=500, detail=e.to_dict())


@router.post(
    "/rag/query",
    response_model=RAGQueryResponse,
    tags=["RAG"],
)
async def rag_query(
    request: RAGQueryRequest,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> RAGQueryResponse:
    """Query the RAG pipeline with a question."""
    try:
        result = await rag_pipeline.query(
            question=request.question,
            system_prompt=request.system_prompt,
            include_sources=request.include_sources,
        )
        return RAGQueryResponse(
            answer=result["answer"],
            has_context=result["has_context"],
            sources=result.get("sources"),
        )
    except ChatbotException as e:
        logger.error("RAG query error", error=str(e))
        raise HTTPException(status_code=500, detail=e.to_dict())
