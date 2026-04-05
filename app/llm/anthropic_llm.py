import json
from typing import Any, AsyncIterator, Optional

from anthropic import AsyncAnthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.exceptions import LLMConnectionError, LLMRateLimitError, LLMResponseError
from app.core.logging import get_logger
from app.llm.base import BaseLLM, LLMResponse, Message, MessageRole, ToolCall, ToolDefinition

logger = get_logger(__name__)


class AnthropicLLM(BaseLLM):
    """Anthropic LLM implementation."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-opus-20240229",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, temperature, max_tokens, **kwargs)
        self.client = AsyncAnthropic(api_key=api_key)
        logger.info("Anthropic LLM initialized", model=model)
    
    def _convert_messages(
        self, messages: list[Message]
    ) -> list[dict[str, Any]]:
        """Convert internal messages to Anthropic format."""
        anthropic_messages: list[dict[str, Any]] = []
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                # System messages handled separately in Anthropic API
                continue
            
            if msg.role == MessageRole.TOOL:
                anthropic_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": msg.content,
                        }
                    ],
                })
            elif msg.tool_calls:
                # Assistant message with tool calls
                content: list[dict[str, Any]] = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    })
                anthropic_messages.append({"role": "assistant", "content": content})
            else:
                anthropic_messages.append({
                    "role": msg.role.value,
                    "content": msg.content,
                })
        
        return anthropic_messages
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        reraise=True,
    )
    async def generate(
        self,
        messages: list[Message],
        system_prompt: Optional[str] = None,
        tools: Optional[list[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response using Anthropic."""
        try:
            anthropic_messages = self._convert_messages(messages)
            
            request_kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": anthropic_messages,
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
            }
            
            if system_prompt:
                request_kwargs["system"] = system_prompt
            
            if tools:
                request_kwargs["tools"] = [t.to_anthropic_format() for t in tools]
            
            response = await self.client.messages.create(**request_kwargs)
            
            # Parse response content
            content_text = ""
            tool_calls = []
            
            for block in response.content:
                if block.type == "text":
                    content_text += block.text
                elif block.type == "tool_use":
                    tool_calls.append(
                        ToolCall(
                            id=block.id,
                            name=block.name,
                            arguments=block.input if isinstance(block.input, dict) else {},
                        )
                    )
            
            return LLMResponse(
                content=content_text,
                tool_calls=tool_calls if tool_calls else None,
                finish_reason=response.stop_reason or "end_turn",
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                },
                model=response.model,
                raw_response=response,
            )
            
        except Exception as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg:
                raise LLMRateLimitError(f"Anthropic rate limit exceeded: {e}", original_error=e)
            elif "connection" in error_msg or "timeout" in error_msg:
                raise LLMConnectionError(f"Anthropic connection error: {e}", original_error=e)
            else:
                raise LLMResponseError(f"Anthropic error: {e}", original_error=e)
    
    async def generate_stream(
        self,
        messages: list[Message],
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Generate a streaming response using Anthropic."""
        try:
            anthropic_messages = self._convert_messages(messages)
            
            request_kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": anthropic_messages,
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
            }
            
            if system_prompt:
                request_kwargs["system"] = system_prompt
            
            async with self.client.messages.stream(**request_kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            logger.error("Anthropic streaming error", error=str(e))
            raise LLMResponseError(f"Anthropic streaming error: {e}", original_error=e)
    
    async def health_check(self) -> bool:
        """Check if Anthropic is accessible."""
        try:
            # Simple message to verify connectivity
            await self.client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return True
        except Exception as e:
            logger.warning("Anthropic health check failed", error=str(e))
            return False
