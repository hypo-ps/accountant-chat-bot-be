import asyncio
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

from app.core.config import settings
from app.core.exceptions import AgentError, AgentTimeoutError
from app.core.logging import get_logger
from app.llm.base import BaseLLM, LLMResponse, Message, MessageRole, ToolCall
from app.agent.tools import ToolRegistry, ToolResult
from app.agent.conversation import Conversation, ConversationManager

logger = get_logger(__name__)


@dataclass
class AgentResponse:
    """Response from the agent."""
    content: str
    conversation_id: str
    tool_calls_made: list[dict[str, Any]]
    usage: dict[str, int]
    iterations: int
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "conversation_id": self.conversation_id,
            "tool_calls_made": self.tool_calls_made,
            "usage": self.usage,
            "iterations": self.iterations,
        }


class Agent:
    """
    Agentic chatbot that can use tools and maintain conversations.
    
    The agent uses an iterative approach:
    1. Send user message to LLM
    2. If LLM requests tool calls, execute them
    3. Send tool results back to LLM
    4. Repeat until LLM provides final response
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        tool_registry: Optional[ToolRegistry] = None,
        conversation_manager: Optional[ConversationManager] = None,
        max_iterations: int = 10,
        timeout_seconds: int = 120,
    ) -> None:
        """
        Initialize the agent.
        
        Args:
            llm: LLM instance for generation
            tool_registry: Registry of available tools
            conversation_manager: Manager for conversations
            max_iterations: Maximum tool calling iterations
            timeout_seconds: Timeout for agent execution
        """
        self.llm = llm
        self.tool_registry = tool_registry or ToolRegistry()
        self.conversation_manager = conversation_manager or ConversationManager(
            default_system_prompt=settings.default_system_prompt
        )
        self.max_iterations = max_iterations or settings.agent_max_iterations
        self.timeout_seconds = timeout_seconds or settings.agent_timeout_seconds
        
        logger.info(
            "Agent initialized",
            max_iterations=self.max_iterations,
            timeout=self.timeout_seconds,
        )
    
    async def chat(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        use_tools: bool = True,
    ) -> AgentResponse:
        """
        Process a chat message.
        
        Args:
            message: User's message
            conversation_id: ID of existing conversation (or create new)
            system_prompt: Override system prompt for this message
            use_tools: Whether to allow tool usage
            
        Returns:
            AgentResponse with the result
        """
        try:
            return await asyncio.wait_for(
                self._process_message(message, conversation_id, system_prompt, use_tools),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise AgentTimeoutError(
                f"Agent execution timed out after {self.timeout_seconds} seconds"
            )
    
    async def _process_message(
        self,
        message: str,
        conversation_id: Optional[str],
        system_prompt: Optional[str],
        use_tools: bool,
    ) -> AgentResponse:
        """Internal message processing logic."""
        # Get or create conversation
        if conversation_id:
            conversation = self.conversation_manager.get_or_create(
                conversation_id,
                system_prompt=system_prompt,
            )
        else:
            conversation = self.conversation_manager.create(system_prompt=system_prompt)
        
        # Add user message
        conversation.add_user_message(message)
        
        # Prepare tools if enabled
        tools = self.tool_registry.get_definitions() if use_tools else None
        
        total_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        tool_calls_made: list[dict[str, Any]] = []
        iterations = 0
        
        while iterations < self.max_iterations:
            iterations += 1
            
            # Generate response
            response = await self.llm.generate(
                messages=conversation.get_messages(),
                system_prompt=conversation.system_prompt,
                tools=tools if tools else None,
            )
            
            # Accumulate usage
            for key in total_usage:
                total_usage[key] += response.usage.get(key, 0)
            
            # Check if we need to execute tools
            if response.has_tool_calls and use_tools:
                # Add assistant message with tool calls
                conversation.add_message(Message(
                    role=MessageRole.ASSISTANT,
                    content=response.content,
                    tool_calls=response.tool_calls,
                ))
                
                # Execute each tool call
                for tool_call in response.tool_calls:
                    result = await self._execute_tool(tool_call)
                    tool_calls_made.append({
                        "tool": tool_call.name,
                        "arguments": tool_call.arguments,
                        "result": result.to_dict(),
                    })
                    
                    # Add tool result to conversation
                    conversation.add_message(Message(
                        role=MessageRole.TOOL,
                        content=result.output,
                        tool_call_id=tool_call.id,
                    ))
            else:
                # Final response - no tool calls
                conversation.add_assistant_message(response.content)
                
                return AgentResponse(
                    content=response.content,
                    conversation_id=conversation.id,
                    tool_calls_made=tool_calls_made,
                    usage=total_usage,
                    iterations=iterations,
                )
        
        raise AgentError(
            f"Agent exceeded maximum iterations ({self.max_iterations})",
            details={"tool_calls_made": tool_calls_made},
        )
    
    async def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call."""
        logger.info("Executing tool", tool=tool_call.name, args=tool_call.arguments)
        return await self.tool_registry.execute(tool_call.name, **tool_call.arguments)
    
    async def chat_stream(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """
        Stream a chat response (without tools).
        
        Note: Streaming does not support tool calling.
        """
        conversation = (
            self.conversation_manager.get_or_create(conversation_id, system_prompt)
            if conversation_id
            else self.conversation_manager.create(system_prompt=system_prompt)
        )
        
        conversation.add_user_message(message)
        
        full_response = ""
        async for chunk in self.llm.generate_stream(
            messages=conversation.get_messages(),
            system_prompt=conversation.system_prompt,
        ):
            full_response += chunk
            yield chunk
        
        conversation.add_assistant_message(full_response)
