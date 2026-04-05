"""
RAG-enabled agent that intelligently decides when to use document retrieval.

This agent has access to the knowledge base via tools and will automatically
search for relevant documents when it determines the query requires it.
"""

from typing import Any, Optional

from app.agent.agent import Agent, AgentResponse
from app.agent.conversation import ConversationManager
from app.agent.tools import ToolRegistry
from app.agent.rag_tools import SearchKnowledgeBaseTool, GetDocumentTool
from app.core.config import settings
from app.core.logging import get_logger
from app.llm.base import BaseLLM
from app.rag.retriever import Retriever

logger = get_logger(__name__)


# Enhanced system prompt that guides the agent on when to use RAG
RAG_AGENT_SYSTEM_PROMPT = """You are a helpful AI assistant with access to a knowledge base of documents.

IMPORTANT GUIDELINES:

1. **Use the search_knowledge_base tool** when:
   - User asks about company policies, procedures, or internal information
   - User references specific documents or manuals
   - User asks questions like "What does our policy say about..." or "According to the guidelines..."
   - You need accurate, up-to-date information that may be in the documents
   - User asks about products, services, or company-specific details

2. **Answer directly WITHOUT using tools** when:
   - User asks general knowledge questions (math, history, science)
   - User is having casual conversation ("Hello", "How are you?")
   - User asks for creative help (writing, brainstorming)
   - User asks something you can confidently answer from your training

3. **When you use the knowledge base**:
   - Cite the source document in your response
   - If no relevant documents are found, say so clearly
   - Don't make up information that isn't in the documents

4. **Be helpful and conversational** while being accurate and grounded in the documents when needed.
"""


class RAGAgent(Agent):
    """
    An agent that can intelligently use RAG when needed.
    
    This extends the base Agent with RAG tools pre-registered,
    allowing the LLM to decide when to search the knowledge base.
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        retriever: Retriever,
        tool_registry: Optional[ToolRegistry] = None,
        conversation_manager: Optional[ConversationManager] = None,
        max_iterations: int = 10,
        timeout_seconds: int = 120,
        include_rag_tools: bool = True,
        custom_system_prompt: Optional[str] = None,
    ) -> None:
        """
        Initialize the RAG-enabled agent.
        
        Args:
            llm: The language model to use
            retriever: The retriever for knowledge base access
            tool_registry: Optional custom tool registry
            conversation_manager: Optional conversation manager
            max_iterations: Max tool calling iterations
            timeout_seconds: Timeout for agent execution
            include_rag_tools: Whether to automatically add RAG tools
            custom_system_prompt: Custom system prompt (will be combined with RAG prompt)
        """
        # Create tool registry if not provided
        if tool_registry is None:
            tool_registry = ToolRegistry()
        
        # Register RAG tools
        if include_rag_tools:
            tool_registry.register(SearchKnowledgeBaseTool(retriever))
            tool_registry.register(GetDocumentTool(retriever))
            logger.info("RAG tools registered with agent")
        
        # Create conversation manager with RAG-aware system prompt
        if conversation_manager is None:
            system_prompt = RAG_AGENT_SYSTEM_PROMPT
            if custom_system_prompt:
                system_prompt = f"{custom_system_prompt}\n\n{RAG_AGENT_SYSTEM_PROMPT}"
            
            conversation_manager = ConversationManager(
                default_system_prompt=system_prompt
            )
        
        # Initialize parent Agent
        super().__init__(
            llm=llm,
            tool_registry=tool_registry,
            conversation_manager=conversation_manager,
            max_iterations=max_iterations,
            timeout_seconds=timeout_seconds,
        )
        
        self.retriever = retriever
        logger.info("RAG Agent initialized")
    
    async def chat(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        use_tools: bool = True,  # RAG tools enabled by default
    ) -> AgentResponse:
        """
        Process a chat message with intelligent RAG usage.
        
        The agent will automatically decide whether to search the
        knowledge base based on the query content.
        
        Args:
            message: User's message
            conversation_id: Existing conversation ID
            system_prompt: Override system prompt (combined with RAG prompt)
            use_tools: Whether to allow tool usage (including RAG)
            
        Returns:
            AgentResponse with the result
        """
        # Combine custom system prompt with RAG prompt if provided
        if system_prompt:
            system_prompt = f"{system_prompt}\n\n{RAG_AGENT_SYSTEM_PROMPT}"
        
        return await super().chat(
            message=message,
            conversation_id=conversation_id,
            system_prompt=system_prompt,
            use_tools=use_tools,
        )


async def create_rag_agent(
    llm: BaseLLM,
    retriever: Retriever,
    custom_tools: Optional[list] = None,
    system_prompt: Optional[str] = None,
) -> RAGAgent:
    """
    Factory function to create a RAG-enabled agent.
    
    Args:
        llm: Language model instance
        retriever: Retriever with access to vector DB
        custom_tools: Additional tools to register
        system_prompt: Custom system prompt to prepend
        
    Returns:
        Configured RAGAgent instance
    """
    tool_registry = ToolRegistry()
    
    # Register custom tools if provided
    if custom_tools:
        for tool in custom_tools:
            tool_registry.register(tool)
    
    return RAGAgent(
        llm=llm,
        retriever=retriever,
        tool_registry=tool_registry,
        custom_system_prompt=system_prompt,
    )
